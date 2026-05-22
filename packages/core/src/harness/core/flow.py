"""Decorator-driven workflow DAG for composing multi-step agent pipelines.

Inspired by CrewAI Flows. Steps are plain async methods; decorators declare
their execution order. :class:`FlowRunner` builds the DAG at construction time
and executes it, threading ``self.state`` through every step.

Example::

    from pydantic import BaseModel
    from harness.core.flow import Flow, FlowRunner, listen, persist, router, start

    class ResearchState(BaseModel):
        topic: str = ""
        outline: str = ""
        draft: str = ""

    class ResearchFlow(Flow[ResearchState]):
        @start
        async def gather_topic(self):
            self.state.topic = "agent frameworks"

        @listen(gather_topic)
        @persist
        async def write_outline(self):
            self.state.outline = f"Outline for: {self.state.topic}"

        @listen(write_outline)
        @router()
        async def decide_depth(self) -> str:
            return "deep" if len(self.state.topic) > 5 else "shallow"

        @listen("deep")
        async def write_long_draft(self):
            self.state.draft = f"[long] {self.state.outline}"

        @listen("shallow")
        async def write_short_draft(self):
            self.state.draft = f"[short] {self.state.outline}"

    store = InMemoryCheckpointStore()
    flow = ResearchFlow()
    runner = FlowRunner(flow, checkpoint_store=store, flow_id="run-1")
    final_state = await runner.run()

    # Fork from the write_outline checkpoint
    cp = await store.load("run-1", "write_outline")
    forked = FlowRunner.from_checkpoint(cp, ResearchFlow())
    alt_state = await forked.run()
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel

if TYPE_CHECKING:
    from harness.core.flow_checkpoint import CheckpointStore, FlowCheckpoint

StateT = TypeVar("StateT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def start(fn: Callable) -> Callable:
    """Mark this method as a flow entry point."""
    fn._flow_start = True  # type: ignore[attr-defined]
    return fn


def listen(target: Callable | str) -> Callable[[Callable], Callable]:
    """Mark this method to run after *target* completes.

    *target* may be a method reference (resolved to its ``__name__``) or a
    plain string label (used for router output routing).
    """

    def decorator(fn: Callable) -> Callable:
        fn._flow_listen = target.__name__ if callable(target) else target  # type: ignore[attr-defined]
        return fn

    return decorator


def router() -> Callable[[Callable], Callable]:
    """Mark this method as a routing step.

    The method must return a string label. Methods decorated with
    ``@listen("that_label")`` will run next.
    """

    def decorator(fn: Callable) -> Callable:
        fn._flow_router = True  # type: ignore[attr-defined]
        return fn

    return decorator


def persist(fn: Callable) -> Callable:
    """Snapshot flow state after this step completes.

    When :class:`FlowRunner` is configured with a ``checkpoint_store``, it
    serialises the current state into a :class:`~harness.core.flow_checkpoint.FlowCheckpoint`
    after every ``@persist``-decorated step. The checkpoint can later be passed
    to :meth:`FlowRunner.from_checkpoint` to fork a new run from that point.
    """
    fn._flow_persist = True  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# Flow base class
# ---------------------------------------------------------------------------


class Flow(Generic[StateT]):
    """Base class for decorator-driven workflows.

    Subclass with a concrete state type::

        class MyFlow(Flow[MyState]):
            ...

    The ``StateT`` is extracted from ``__orig_bases__`` at class creation time.
    If you need to pass an initial state, do so at construction::

        flow = MyFlow(state=MyState(field="value"))
    """

    state: StateT
    _state_type: type  # set by __init_subclass__

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", []):
            if get_origin(base) is Flow:
                args = get_args(base)
                if args:
                    cls._state_type = args[0]
                    break

    def __init__(self, *, state: StateT | None = None) -> None:
        state_type: type | None = getattr(self.__class__, "_state_type", None)
        if state is not None:
            self.state = state
        elif state_type is not None:
            self.state = state_type()  # type: ignore[assignment]
        else:
            raise TypeError(
                f"{type(self).__name__} must be parameterized as Flow[YourState]"
                " or pass state= at construction"
            )


# ---------------------------------------------------------------------------
# FlowRunner
# ---------------------------------------------------------------------------


class FlowRunner:
    """Execute a :class:`Flow` DAG built from decorator metadata.

    The runner inspects all decorated methods on the flow instance at
    construction time, builds an adjacency dict, then walks it in BFS order
    during :meth:`run`.

    Optional checkpoint support::

        runner = FlowRunner(flow, checkpoint_store=store, flow_id="run-42")
        await runner.run()  # saves a checkpoint after every @persist step

    Fork from a past checkpoint::

        cp = await store.load("run-42", "expensive_step")
        forked = FlowRunner.from_checkpoint(cp, MyFlow())
        await forked.run()  # re-runs from after "expensive_step"
    """

    def __init__(
        self,
        flow: Flow,
        *,
        checkpoint_store: CheckpointStore | None = None,
        flow_id: str | None = None,
        _resume_from: str | None = None,
    ) -> None:
        self._flow = flow
        self._checkpoint_store = checkpoint_store
        self._flow_id = flow_id
        self._resume_from = _resume_from
        self._steps: dict[str, Callable] = {}
        self._starts: list[str] = []
        self._routers: set[str] = set()
        self._persists: set[str] = set()
        # target_name → [listener_names]  (used for both method and label routing)
        self._listens: dict[str, list[str]] = {}
        self._build()

    def _build(self) -> None:
        flow_cls = type(self._flow)
        for name in dir(flow_cls):
            method = getattr(flow_cls, name, None)
            if method is None or not callable(method):
                continue
            fn = getattr(method, "__func__", method)

            is_start = getattr(fn, "_flow_start", False)
            listen_target = getattr(fn, "_flow_listen", None)
            is_router = getattr(fn, "_flow_router", False)
            is_persist = getattr(fn, "_flow_persist", False)

            if is_start or listen_target is not None or is_router:
                self._steps[name] = getattr(self._flow, name)

            if is_start:
                self._starts.append(name)
            if listen_target is not None:
                self._listens.setdefault(listen_target, []).append(name)
            if is_router:
                self._routers.add(name)
            if is_persist:
                self._persists.add(name)

    async def run(self) -> StateT:
        """Execute the flow DAG and return the final state.

        If ``_resume_from`` is set (via :meth:`from_checkpoint`), the named
        step is treated as already executed and its listeners are queued first.
        Otherwise, entry points (``@start``) are queued.

        Each step runs exactly once per ``run()`` call. After every
        ``@persist``-decorated step, the current state is saved to the
        configured ``checkpoint_store`` (if any).
        """
        executed: set[str] = set()

        if self._resume_from:
            executed.add(self._resume_from)
            queue: list[str] = list(self._listens.get(self._resume_from, []))
        else:
            queue = list(self._starts)

        while queue:
            step_name = queue.pop(0)
            if step_name in executed:
                continue
            executed.add(step_name)

            step_fn = self._steps.get(step_name)
            if step_fn is None:
                continue

            if inspect.iscoroutinefunction(step_fn):
                result = await step_fn()
            else:
                result = step_fn()

            if step_name in self._persists and self._checkpoint_store is not None:
                await self._save_checkpoint(step_name)

            if step_name in self._routers and isinstance(result, str):
                # Router: resolve label → listeners
                for listener_name in self._listens.get(result, []):
                    if listener_name not in executed:
                        queue.append(listener_name)
            else:
                # Method-based edges: step_name → @listen(step_name)
                for listener_name in self._listens.get(step_name, []):
                    if listener_name not in executed:
                        queue.append(listener_name)

        return self._flow.state  # type: ignore[return-value]

    async def _save_checkpoint(self, step_name: str) -> None:
        from harness.core.flow_checkpoint import FlowCheckpoint

        flow_id = self._flow_id or f"flow_{uuid.uuid4().hex[:12]}"
        self._flow_id = flow_id
        checkpoint = FlowCheckpoint(
            flow_id=flow_id,
            step_name=step_name,
            state_json=self._flow.state.model_dump_json(),
            created_at=datetime.now(UTC),
        )
        assert self._checkpoint_store is not None
        await self._checkpoint_store.save(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: FlowCheckpoint,
        flow: Flow,
        *,
        checkpoint_store: CheckpointStore | None = None,
    ) -> FlowRunner:
        """Return a :class:`FlowRunner` with state restored from *checkpoint*.

        Execution resumes from the listeners of the persisted step — i.e. the
        step itself is NOT re-run, only what comes after it. This is the
        "fork" semantics: take the world as it was at that point and continue.

        Example::

            cp = await store.load("run-42", "write_outline")
            forked = FlowRunner.from_checkpoint(cp, ResearchFlow())
            state = await forked.run()
        """
        state_type = type(flow.state)
        flow.state = state_type.model_validate_json(checkpoint.state_json)
        return cls(
            flow,
            checkpoint_store=checkpoint_store,
            flow_id=checkpoint.flow_id,
            _resume_from=checkpoint.step_name,
        )


__all__ = ["Flow", "FlowRunner", "listen", "persist", "router", "start"]

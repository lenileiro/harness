"""Decorator-driven workflow DAG for composing multi-step agent pipelines.

Inspired by CrewAI Flows. Steps are plain async methods; decorators declare
their execution order. :class:`FlowRunner` builds the DAG at construction time
and executes it, threading ``self.state`` through every step.

Example::

    from pydantic import BaseModel
    from harness.core.flow import Flow, FlowRunner, listen, router, start

    class ResearchState(BaseModel):
        topic: str = ""
        outline: str = ""
        draft: str = ""

    class ResearchFlow(Flow[ResearchState]):
        @start
        async def gather_topic(self):
            self.state.topic = "agent frameworks"

        @listen(gather_topic)
        async def write_outline(self):
            self.state.outline = f"Outline for: {self.state.topic}"

        @router()
        async def decide_depth(self) -> str:
            return "deep" if len(self.state.topic) > 5 else "shallow"

        @listen("deep")
        async def write_long_draft(self):
            self.state.draft = f"[long] {self.state.outline}"

        @listen("shallow")
        async def write_short_draft(self):
            self.state.draft = f"[short] {self.state.outline}"

    flow = ResearchFlow()
    runner = FlowRunner(flow)
    final_state = await runner.run()

Note:
    ``@persist`` (snapshot/fork-from-checkpoint) is not yet implemented.
    State lives in memory only; a crash loses it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel

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
    """

    def __init__(self, flow: Flow) -> None:
        self._flow = flow
        self._steps: dict[str, Callable] = {}
        self._starts: list[str] = []
        self._routers: set[str] = set()
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

            if is_start or listen_target is not None or is_router:
                self._steps[name] = getattr(self._flow, name)

            if is_start:
                self._starts.append(name)
            if listen_target is not None:
                self._listens.setdefault(listen_target, []).append(name)
            if is_router:
                self._routers.add(name)

    async def run(self) -> StateT:
        """Execute the flow DAG and return the final state.

        Entry points (``@start``) are queued first. Execution proceeds in BFS
        order, respecting ``@listen`` edges and ``@router`` label routing.
        Each step runs exactly once per ``run()`` call.
        """
        executed: set[str] = set()
        queue: list[str] = list(self._starts)

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


__all__ = ["Flow", "FlowRunner", "listen", "router", "start"]

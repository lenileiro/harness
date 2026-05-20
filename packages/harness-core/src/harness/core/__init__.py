"""Harness core: protocols, schemas, and runtime."""

from harness.core.adapter import Adapter
from harness.core.errors import (
    ApprovalDeniedError,
    CancelledError,
    ConfigurationError,
    HarnessError,
    InternalError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
    ToolError,
)
from harness.core.events import (
    Done,
    ErrorEvent,
    Event,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from harness.core.failover import ErrorKind, FailoverPolicy, classify
from harness.core.planner import NoOpPlanner, Plan, PlanContext, Planner, PlanStep
from harness.core.runtime import Agent
from harness.core.schemas import (
    ApprovalDecision,
    Capabilities,
    Message,
    Role,
    RunRequest,
    Session,
    SessionStatus,
    ToolCall,
    ToolResult,
    Usage,
)
from harness.core.storage import Storage
from harness.core.telemetry import configure_logging, get_logger, span
from harness.core.tools import (
    ApprovalHandler,
    ApprovalPolicy,
    AutoApprove,
    AutoDeny,
    Tool,
    ToolRegistry,
)

__version__ = "0.0.0"

__all__ = [
    "Adapter",
    "Agent",
    "ApprovalDecision",
    "ApprovalDeniedError",
    "ApprovalHandler",
    "ApprovalPolicy",
    "AutoApprove",
    "AutoDeny",
    "CancelledError",
    "Capabilities",
    "ConfigurationError",
    "Done",
    "ErrorEvent",
    "ErrorKind",
    "Event",
    "FailoverPolicy",
    "HarnessError",
    "InternalError",
    "Message",
    "ModelUnavailableError",
    "NetworkError",
    "NoOpPlanner",
    "Plan",
    "PlanContext",
    "PlanStep",
    "Planner",
    "RateLimitError",
    "Role",
    "RunRequest",
    "Session",
    "SessionStatus",
    "StepCompleted",
    "StepStarted",
    "Storage",
    "TextDelta",
    "TimeoutError",
    "Tool",
    "ToolCall",
    "ToolCallEvent",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolResultEvent",
    "Usage",
    "__version__",
    "classify",
    "configure_logging",
    "get_logger",
    "span",
]

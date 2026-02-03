"""Debate module exports.

AutoGen is treated as an optional dependency so the LangGraph-style engine can run
in minimal environments (e.g., offline unit tests) without requiring pyautogen.
"""

import warnings

# Silence noisy optional-dependency warnings that may be triggered when importing AutoGen.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"flaml(\..*)?",
    message=r"flaml\.automl is not available\..*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"autogen\.oai\.gemini(\..*)?",
    message=r"\s*All support for the `google\.generativeai` package has ended\..*",
)

from debate.langgraph_coordinator import LangGraphDebateCoordinator, GraphDebateResult

try:
    from debate.autogen_coordinator import AutoGenDebateCoordinator, DebateResult
except Exception:  # optional dependency (pyautogen -> autogen)
    AutoGenDebateCoordinator = None  # type: ignore
    DebateResult = None  # type: ignore

__all__ = [
    'LangGraphDebateCoordinator',
    'GraphDebateResult',
    'AutoGenDebateCoordinator',
    'DebateResult',
]

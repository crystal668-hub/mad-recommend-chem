"""
经验模块初始化文件
"""

import warnings

# Silence noisy optional-dependency warnings that may be triggered when importing ExperienceExtractor.
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

from experience.experience_store import ExperienceStore

# ExperienceExtractor depends on optional multi-agent frameworks (e.g. autogen).
# Keep the package importable even when those dependencies are not installed.
try:
    from experience.experience_extractor import ExperienceExtractor
except Exception:  # pragma: no cover
    ExperienceExtractor = None  # type: ignore

__all__ = ['ExperienceStore', 'ExperienceExtractor']

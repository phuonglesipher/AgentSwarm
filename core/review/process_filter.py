from __future__ import annotations

import re

# ---- Base keywords shared by ALL review profiles ----
BASE_PROCESS_KEYWORDS: tuple[str, ...] = (
    # Organizational / process
    "dri",
    "merge authority",
    "sign off",
    "sign-off",
    "signoff",
    "accountability",
    "named test owner",
    "owner assignment",
    "commit",
    "pull request",
    "pr ",
    "provenance",
    "regression-origin",
    "regression origin",
    "behavioral delta",
    "what changed and where",
    "regression boundary",
    "stakeholder",
    "approval chain",
    "release gate",
    "release sign",
    "qa handoff",
    "qa sign",
    "deployment plan",
    "rollback plan",
    "jira",
    "ticket",
    "sprint",
    "backlog",
    "code review",
    "peer review",
    "change management",
    # Runtime data — agents cannot launch the editor or capture runtime data
    "stat gpu",
    "stat nanite",
    "gpu capture",
    "gpu timestamp",
    "renderdoc",
    "pix capture",
    "gpu insights",
    "frame capture",
    "runtime capture",
    "runtime profil",
    "runtime stats",
    "runtime measurement",
    "r.nanite.showstats",
    "r.nanite.visualize",
    "actual gpu",
    "real gpu",
    "measured gpu",
    "gpu evidence",
    "runtime gpu",
    "launch the editor",
    "run the editor",
    "open the editor",
    "in-editor",
    "play-in-editor",
    "pie session",
    # Hardware-specific (shared)
    "profile in production",
    "shipping build",
    "console devkit",
    "target hardware",
)

# ---- Base regex patterns shared by ALL review profiles ----
BASE_PROCESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"assign\s+(a\s+)?named\b", re.IGNORECASE),
    re.compile(r"before\s+(implementation|deployment|release)\s+starts", re.IGNORECASE),
    re.compile(r"who\s+(is|will be)\s+(responsible|accountable)", re.IGNORECASE),
    re.compile(r"\borganizational\b", re.IGNORECASE),
    # Runtime data patterns
    re.compile(r"\bstat\s+(gpu|nanite|scenerendering)\b", re.IGNORECASE),
    re.compile(r"\br\.nanite\.(showstats|visualize)\b", re.IGNORECASE),
    re.compile(r"\b(gpu|runtime)\s+(capture|timestamp|profil|measurement)", re.IGNORECASE),
    re.compile(r"\b(renderdoc|pix|nsight)\s+(capture|frame|session)", re.IGNORECASE),
    re.compile(r"\brun\s+.*\s+in\s+(the\s+)?editor\b", re.IGNORECASE),
    re.compile(r"\bcapture\s+.*\s+(frame|gpu|profil)", re.IGNORECASE),
    re.compile(r"\blaunch\s+.*\s+editor\b", re.IGNORECASE),
    re.compile(r"\bactual\s+(gpu|frame|runtime)\s+(time|data|metric|number)", re.IGNORECASE),
    re.compile(r"\bmeasured\s+(gpu|frame|baseline|timing)", re.IGNORECASE),
    re.compile(r"\bplay[- ]in[- ]editor\b", re.IGNORECASE),
    re.compile(r"\bpie\s+session\b", re.IGNORECASE),
    re.compile(r"\bin[- ]editor\s+profil", re.IGNORECASE),
)


def is_process_only_feedback(
    text: str,
    *,
    extra_keywords: tuple[str, ...] = (),
    extra_patterns: tuple[re.Pattern[str], ...] = (),
) -> bool:
    """Return True if *text* is a process-only or impossible-for-agent request."""
    lowered = str(text).strip().lower()
    if not lowered:
        return True
    all_keywords = BASE_PROCESS_KEYWORDS + extra_keywords
    if any(kw in lowered for kw in all_keywords):
        return True
    all_patterns = BASE_PROCESS_PATTERNS + extra_patterns
    return any(p.search(text) for p in all_patterns)

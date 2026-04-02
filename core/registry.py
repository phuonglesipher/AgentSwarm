from __future__ import annotations

from collections import Counter
import re
from typing import Any

from core.models import WorkflowMetadata, WorkflowRuntime
from core.text_utils import normalize_text


_ROUTING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "for",
    "from",
    "hello",
    "help",
    "hey",
    "hi",
    "how",
    "i",
    "if",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "tell",
    "that",
    "the",
    "there",
    "this",
    "to",
    "today",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
    "agent",
    "agentswarm",
    "current",
    "final",
    "loop",
    "loops",
    "prompt",
    "prompts",
    "quality",
    "ready",
    "report",
    "request",
    "requests",
    "reviewer",
    "round",
    "rounds",
    "score",
    "scoring",
    "strict",
    "task",
    "tasks",
    "user",
    "users",
    "workflow",
    "workflows",
}
_ROUTING_ACTION_HINTS = {
    "add",
    "analyze",
    "architect",
    "architecture",
    "bug",
    "debug",
    "design",
    "feature",
    "fix",
    "handoff",
    "implement",
    "implementation",
    "improve",
    "investigate",
    "maintenance",
    "optimize",
    "optimization",
    "plan",
    "refactor",
    "regression",
    "repair",
    "research",
    "review",
}
_ROUTING_DOMAIN_HINTS = {
    "ability",
    "architecture",
    "blueprint",
    "bug",
    "build",
    "cause",
    "charge",
    "character",
    "code",
    "combat",
    "compile",
    "callstack",
    "crash",
    "dash",
    "debug",
    "design",
    "dodge",
    "doc",
    "docs",
    "dump",
    "engine",
    "enemy",
    "feature",
    "file",
    "files",
    "freeze",
    "gameplay",
    "graph",
    "hang",
    "investigate",
    "jump",
    "log",
    "logs",
    "mechanic",
    "melee",
    "memory",
    "movement",
    "optimize",
    "optimization",
    "performance",
    "player",
    "project",
    "recharge",
    "refactor",
    "regression",
    "root",
    "runtime",
    "source",
    "sources",
    "spawn",
    "stability",
    "state",
    "tdr",
    "test",
    "tests",
    "tool",
    "tools",
    "traversal",
    "violation",
    "weapon",
}


def _stem_routing_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stemmed = token[:-3]
        if len(stemmed) >= 2 and stemmed[-1] == stemmed[-2]:
            stemmed = stemmed[:-1]
        return stemmed
    if token.endswith("ed") and len(token) > 4:
        stemmed = token[:-2]
        if len(stemmed) >= 2 and stemmed[-1] == stemmed[-2]:
            stemmed = stemmed[:-1]
        return stemmed
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith(("is", "ss", "us")):
        return token[:-1]
    return token


def _routing_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in re.findall(r"[a-z0-9_]+", normalize_text(value or "")):
        token = _stem_routing_token(raw_token)
        if len(token) < 3 or token in _ROUTING_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _workflow_routing_tokens(metadata: WorkflowMetadata) -> set[str]:
    return _routing_tokens(
        " ".join([metadata.name, metadata.qualified_name, metadata.description, *metadata.capabilities])
    )


def _workflow_action_hints(metadata: WorkflowMetadata) -> set[str]:
    normalized_name = normalize_text(metadata.name)
    hints = set(_ROUTING_ACTION_HINTS)
    if "planner" in normalized_name:
        hints |= {"plan", "research", "design", "architecture", "handoff"}
    if "investigation" in normalized_name:
        hints |= {"analyze", "investigate", "optimization", "optimize", "review", "root", "cause"}
    if "engineer" in normalized_name and "planner" not in normalized_name:
        hints |= {"add", "bug", "debug", "feature", "fix", "implement", "maintenance", "refactor", "repair"}
    return {_stem_routing_token(item) for item in hints}


def _workflow_name_mentions(metadata: WorkflowMetadata) -> set[str]:
    normalized_name = normalize_text(metadata.name)
    normalized_qualified_name = normalize_text(metadata.qualified_name)
    mentions = {
        normalized_name,
        normalized_name.replace("-", " "),
        normalized_name.replace("-", ""),
        normalized_qualified_name,
        normalized_qualified_name.replace("-", " "),
    }
    return {item for item in mentions if item}


class WorkflowRegistry:
    def __init__(self, *, preferred_namespaces: tuple[str, ...] = ("project", "agentswarm")) -> None:
        self._workflows: dict[str, WorkflowRuntime] = {}
        self._aliases: dict[str, str] = {}
        self._namespace_priority = {
            namespace: index for index, namespace in enumerate(preferred_namespaces)
        }

    def register(self, runtime: WorkflowRuntime) -> None:
        metadata = runtime.metadata
        self._workflows[metadata.qualified_name] = runtime
        alias_target = self._aliases.get(metadata.name)
        if alias_target is None:
            self._aliases[metadata.name] = metadata.qualified_name
            return

        current_runtime = self._workflows[alias_target]
        current_priority = self._namespace_priority.get(current_runtime.metadata.namespace, 999)
        incoming_priority = self._namespace_priority.get(metadata.namespace, 999)
        if incoming_priority <= current_priority:
            self._aliases[metadata.name] = metadata.qualified_name

    def get(self, name: str) -> WorkflowRuntime:
        qualified_name = self._aliases.get(name, name)
        return self._workflows[qualified_name]

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        runtime = self.get(name)
        return runtime.invoke(payload)

    def list_metadata(self, exposed_only: bool = False, include_shadowed: bool = True) -> list[WorkflowMetadata]:
        if include_shadowed:
            metadata = [runtime.metadata for runtime in self._workflows.values()]
        else:
            metadata = [self._workflows[qualified_name].metadata for qualified_name in self._aliases.values()]
        if exposed_only:
            metadata = [item for item in metadata if item.exposed]
        return sorted(metadata, key=lambda item: item.qualified_name)

    def route(self, task_description: str) -> WorkflowMetadata | None:
        metadata_items = self.list_metadata(exposed_only=True, include_shadowed=False)
        if not metadata_items:
            return None

        task_text = str(task_description or "").strip()
        if not task_text:
            return None

        task_tokens = _routing_tokens(task_text)
        if not task_tokens:
            return None

        has_domain_signal = bool(task_tokens & {_stem_routing_token(item) for item in _ROUTING_DOMAIN_HINTS})
        has_action_signal = bool(task_tokens & {_stem_routing_token(item) for item in _ROUTING_ACTION_HINTS})
        if not has_domain_signal or not has_action_signal:
            return None

        workflow_tokens = {
            metadata.qualified_name: _workflow_routing_tokens(metadata)
            for metadata in metadata_items
        }
        token_document_frequency = Counter(
            token
            for tokens in workflow_tokens.values()
            for token in tokens
        )

        normalized_task_text = normalize_text(task_text)
        best_match: WorkflowMetadata | None = None
        best_score = 0.0
        best_action_fit = False
        for metadata in metadata_items:
            overlap = task_tokens & workflow_tokens[metadata.qualified_name]
            if not overlap:
                continue

            score = sum(1.0 / token_document_frequency[token] for token in overlap)
            action_fit = bool(task_tokens & _workflow_action_hints(metadata))
            if any(mention in normalized_task_text for mention in _workflow_name_mentions(metadata)):
                score += 1.0
                action_fit = True

            if best_match is None or score > best_score or (score == best_score and action_fit and not best_action_fit):
                best_match = metadata
                best_score = score
                best_action_fit = action_fit

        if best_match is None or best_score <= 0 or not best_action_fit:
            return None
        return best_match

    def matches_multiple_workflows(self, text: str, min_matches: int = 2) -> bool:
        """Return True if *text* has significant token overlap with 2+ exposed workflows."""
        metadata_items = self.list_metadata(exposed_only=True, include_shadowed=False)
        text_tokens = _routing_tokens(text)
        if not text_tokens:
            return False

        matched_count = 0
        for metadata in metadata_items:
            wf_tokens = _workflow_routing_tokens(metadata)
            overlap = text_tokens & wf_tokens
            if len(overlap) >= 2:
                matched_count += 1
            if matched_count >= min_matches:
                return True
        return False

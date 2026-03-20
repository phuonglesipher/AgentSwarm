from __future__ import annotations

import json
from pathlib import Path

from .models import ScorePolicy, ScoreSnapshot


def _history_path(artifact_dir: Path, policy: ScorePolicy) -> Path:
    return artifact_dir / policy.history_filename


def _serialize_snapshot(snapshot: ScoreSnapshot) -> dict[str, object]:
    return {
        "system_id": snapshot.system_id,
        "round_index": snapshot.round_index,
        "score": snapshot.score,
        "threshold": snapshot.threshold,
        "max_score": snapshot.max_score,
        "explicit_approval": snapshot.explicit_approval,
        "approved": snapshot.approved,
        "confidence": snapshot.confidence,
        "confidence_label": snapshot.confidence_label,
        "confidence_reason": snapshot.confidence_reason,
        "baseline_score": snapshot.baseline_score,
        "score_delta_from_baseline": snapshot.score_delta_from_baseline,
    }


def _deserialize_snapshot(payload: dict[str, object], policy: ScorePolicy) -> ScoreSnapshot | None:
    if str(payload.get("system_id", "")).strip() != policy.system_id:
        return None
    try:
        return ScoreSnapshot(
            system_id=policy.system_id,
            round_index=max(1, int(payload["round_index"])),
            score=max(0, min(int(payload["score"]), 100)),
            threshold=max(0, int(payload["threshold"])),
            max_score=max(0, int(payload["max_score"])),
            explicit_approval=bool(payload["explicit_approval"]),
            approved=bool(payload["approved"]),
            confidence=None if payload.get("confidence") is None else float(payload["confidence"]),
            confidence_label=str(payload.get("confidence_label", "")).strip() or "unmeasured",
            confidence_reason=str(payload.get("confidence_reason", "")).strip(),
            baseline_score=None if payload.get("baseline_score") is None else int(payload["baseline_score"]),
            score_delta_from_baseline=(
                None
                if payload.get("score_delta_from_baseline") is None
                else int(payload["score_delta_from_baseline"])
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_score_history(artifact_dir: Path, policy: ScorePolicy) -> tuple[ScoreSnapshot, ...]:
    path = _history_path(artifact_dir, policy)
    if not path.exists():
        return ()

    snapshots: list[ScoreSnapshot] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        snapshot = _deserialize_snapshot(payload, policy)
        if snapshot is not None:
            snapshots.append(snapshot)
    snapshots.sort(key=lambda item: item.round_index)
    return tuple(snapshots)


def record_score_snapshot(
    artifact_dir: Path,
    policy: ScorePolicy,
    snapshot: ScoreSnapshot,
) -> tuple[ScoreSnapshot, ...]:
    existing = {item.round_index: item for item in load_score_history(artifact_dir, policy)}
    existing[snapshot.round_index] = snapshot
    ordered = tuple(existing[index] for index in sorted(existing))
    path = _history_path(artifact_dir, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(_serialize_snapshot(item), sort_keys=True) for item in ordered)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
    return ordered

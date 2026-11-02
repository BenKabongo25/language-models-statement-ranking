from __future__ import annotations

from typing import Any

_FAILURE_STATUSES = {"error", "partial"}


def payload_present(stage: int, record: dict[str, Any]) -> bool:
    if stage == 1:
        return bool(record.get("statements"))
    if stage == 2:
        projection = record.get("projection")
        return bool(
            isinstance(projection, dict)
            and str(projection.get("user", "")).strip()
            and str(projection.get("item", "")).strip()
        )
    if stage == 3:
        return record.get("stance") in {"positive", "negative", "neutral"}
    raise ValueError(f"Unsupported stage: {stage}")


def should_recover(stage: int, record: dict[str, Any]) -> bool:
    status = str(record.get("status", "error"))
    if status in _FAILURE_STATUSES:
        return True
    # Also retry structurally successful records with no useful payload.
    return status == "ok" and not payload_present(stage, record)


def recovery_succeeded(stage: int, record: dict[str, Any]) -> bool:
    return record.get("status") == "ok" and payload_present(stage, record)


def _status_rank(status: str) -> int:
    return {"error": 0, "partial": 2, "ok": 3}.get(status, 0)


def _payload_size(stage: int, record: dict[str, Any]) -> int:
    if stage == 1:
        return len(record.get("statements") or [])
    return 1 if payload_present(stage, record) else 0


def quality_key(stage: int, record: dict[str, Any]) -> tuple[int, int, int]:
    # Non-empty evidence dominates an empty record; within that, prefer a cleaner
    # status and then more valid Stage-1 statements.
    return (
        1 if payload_present(stage, record) else 0,
        _status_rank(str(record.get("status", "error"))),
        _payload_size(stage, record),
    )


def choose_record(stage: int, primary: dict[str, Any], recovered: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if recovered is None:
        return primary, "primary"
    if quality_key(stage, recovered) > quality_key(stage, primary):
        return recovered, "recovery"
    return primary, "primary"


def target_policy_name() -> str:
    return "status-in-error-partial-or-ok-with-empty-payload"

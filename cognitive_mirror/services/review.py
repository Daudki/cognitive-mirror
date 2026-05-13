"""Review queue service for collecting opt-in interaction cases.

Stores pending cases for human review and moves approved cases to an
approved corpus used for offline fine-tuning. Storage is file-based JSONL
to keep dependencies minimal.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


BASE = Path(__file__).resolve().parents[2]
CASES_DIR = BASE / "data" / "interaction_cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)

PENDING_FILE = CASES_DIR / "pending.jsonl"
APPROVED_FILE = CASES_DIR / "approved.jsonl"


def _append_line(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def submit_case(case: Dict[str, Any]) -> None:
    """Submit a case to the pending review queue.

    The case should include: text, emotion, sentiment, mind_state, consent (bool), metadata.
    """
    envelope = {
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        **case,
    }
    _append_line(PENDING_FILE, envelope)


def list_pending() -> List[Dict[str, Any]]:
    return _read_lines(PENDING_FILE)


def list_approved() -> List[Dict[str, Any]]:
    return _read_lines(APPROVED_FILE)


def approve_case(index: int) -> Optional[Dict[str, Any]]:
    """Approve a case by its index in the pending list; move it to approved.

    Returns the approved case or None if index invalid.
    """
    pending = _read_lines(PENDING_FILE)
    if index < 0 or index >= len(pending):
        return None
    case = pending.pop(index)
    # Write back pending
    with PENDING_FILE.open("w", encoding="utf-8") as f:
        for item in pending:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Append to approved
    _append_line(APPROVED_FILE, {
        "approved_at": datetime.utcnow().isoformat() + "Z",
        **case,
    })
    return case

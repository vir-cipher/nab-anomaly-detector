"""Project metadata loader — reads .project-meta/ at runtime."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / ".project-meta"


def load_plan():
    """Return the frozen plan (Rule 1 — read-only after bootstrap)."""
    return json.loads((META / "plan.json").read_text(encoding="utf-8"))


def load_ledger():
    """Return the canonical ledger (Rule 5)."""
    return json.loads((META / "ledger.json").read_text(encoding="utf-8"))


def load_spec():
    """Return the frozen spec as a string."""
    return (META / "spec.md").read_text(encoding="utf-8")


def project_root():
    """Return the repo root Path."""
    return ROOT

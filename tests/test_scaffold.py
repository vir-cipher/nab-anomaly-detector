"""Scaffold tests — verify the project skeleton is sound (Rule 2).

These tests check the project itself: README length, frozen plan order,
no secret-shaped strings, CI workflow present. They run on every push
so the rules are enforced by code, not memory.
"""
import json
import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def test_readme_exists_and_length():
    readme = ROOT / "README.md"
    assert readme.exists(), "README.md missing"
    wc = _word_count(readme)
    assert wc >= 200, f"README.md has {wc} words, need >= 200"


def test_walkthrough_exists_and_length():
    wt = ROOT / "docs" / "WALKTHROUGH.md"
    assert wt.exists(), "docs/WALKTHROUGH.md missing"
    wc = _word_count(wt)
    assert wc >= 300, f"WALKTHROUGH.md has {wc} words, need >= 300"


def test_spec_exists():
    assert (ROOT / ".project-meta" / "spec.md").exists(), "spec.md missing"


def test_plan_exists_and_has_21_steps():
    plan_path = ROOT / ".project-meta" / "plan.json"
    assert plan_path.exists(), "plan.json missing"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["steps"]) == 21, f"plan has {len(plan['steps'])} steps, expected 21"


def test_plan_order_immutable():
    """Step IDs must stay in their original order (Rule 1)."""
    plan_path = ROOT / ".project-meta" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    ids = [s["id"] for s in plan["steps"]]
    expected = [f"step-{i:03d}" for i in range(21)]
    assert ids == expected, f"Plan step order changed! Got {ids[:5]}..."
    # Frozen hash — if this breaks, the plan was tampered with.
    digest = hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]
    FROZEN_HASH = "b0b tried to ch"  # placeholder until computed
    # For now enforce structural order; hash check added after bootstrap.


def test_no_secrets():
    """Grep for secret-shaped strings (Rule 4)."""
    patterns = [
        r"ANTHROPIC_API_KEY\s*=",
        r"sk-[a-zA-Z0-9]{20,}",
        r"password\s*=\s*['\"][^'\"]+['\"]",
    ]
    for root_dir, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for fname in files:
            if fname.endswith((".py", ".json", ".yml", ".yaml", ".md", ".txt")):
                fpath = Path(root_dir) / fname
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                for pat in patterns:
                    assert not re.search(pat, content), (
                        f"Secret pattern '{pat}' found in {fpath.relative_to(ROOT)}"
                    )


def test_ci_workflow_exists():
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci.exists(), ".github/workflows/ci.yml missing"


def test_required_directories():
    for d in ["src", "data", "results", "docs", ".project-meta", "tests"]:
        assert (ROOT / d).is_dir(), f"Directory '{d}' missing"


def test_ledger_exists_and_valid():
    ledger_path = ROOT / ".project-meta" / "ledger.json"
    assert ledger_path.exists(), "ledger.json missing"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["project"] == "nab-anomaly-detector"
    assert ledger["repo"] == "vir-cipher/nab-anomaly-detector"
    assert ledger["author"] == "vir-cipher"


def test_meta_loader():
    """Verify src/meta.py can load all metadata files."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from meta import load_plan, load_ledger, load_spec, project_root
    plan = load_plan()
    assert len(plan["steps"]) == 21
    ledger = load_ledger()
    assert ledger["project"] == "nab-anomaly-detector"
    spec = load_spec()
    assert "Frozen Specification" in spec
    assert project_root() == ROOT

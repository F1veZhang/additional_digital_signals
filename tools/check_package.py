"""Data-free syntax and manifest check; never import a research module."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "docs/source_code_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        p = ROOT / item["package_relative"]
        if not p.is_file():
            failures.append({"path": item["package_relative"], "issue": "missing"})
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != item["package_sha256"]:
            failures.append({"path": item["package_relative"], "issue": "hash changed"})
    python_files = sorted(ROOT.rglob("*.py"))
    for p in python_files:
        try:
            with tokenize.open(p) as f:
                ast.parse(f.read(), filename=str(p))
        except (SyntaxError, UnicodeError) as exc:
            failures.append({"path": str(p.relative_to(ROOT)), "issue": str(exc)})
    forbidden = {".csv", ".tsv", ".parquet", ".npz", ".npy", ".pkl", ".pickle", ".gz", ".zip", ".pt", ".pth", ".cbm", ".jsonl", ".log", ".docx", ".pdf"}
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in forbidden and ".git" not in p.parts:
            failures.append({"path": str(p.relative_to(ROOT)), "issue": "unexpected data/binary/log file"})
    print(json.dumps({"status": "pass" if not failures else "failed", "research_source_files": len(manifest["files"]), "python_files_parsed": len(python_files), "research_code_imported": False, "data_loaded": False, "analysis_run": False, "failures": failures}, indent=2))
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())

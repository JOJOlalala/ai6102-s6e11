"""Print runs/log.jsonl as a markdown table (best CV first) and save it to runs/summary.md.

Usage (project root): .venv/bin/python -I src/summarize.py [runs]
"""
import json
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
recs = {}
for line in (runs / "log.jsonl").read_text().splitlines():
    r = json.loads(line)
    recs[r["name"]] = r  # a rerun with the same name keeps the latest result
rows = sorted(recs.values(), key=lambda r: -r["cv_auc"])
out = ["| CV AUC | fold sd | run | features | seconds |", "|---:|---:|---|---:|---:|"]
for r in rows:
    out.append(f"| {r['cv_auc']:.5f} | {r['fold_sd']:.5f} | {r['name']} | {r.get('n_features', '')} | {r['seconds']} |")
text = "\n".join(out)
(runs / "summary.md").write_text(text + "\n")
print(text)

import argparse
import json
from pathlib import Path
from app.annotations import load_annotations

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("annotations", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    rows = load_annotations(a.annotations, manifest)
    if a.output.resolve() in (a.annotations.resolve(), a.manifest.resolve()):
        p.error("Output must not overwrite input files")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    print(f"Validated {len(rows)} event annotations. No training started.")

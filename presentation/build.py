"""Merge presentation/sections/NN_*.ipynb, in filename order, into concept_transformer_presentation.ipynb.

Run from anywhere: python presentation/build.py [--force]
Only nbformat is used; no notebook code is executed. The build refuses to overwrite a merged
notebook that already holds outputs (an executed HiPerGator run) unless --force is given.
"""
from pathlib import Path
import sys
import uuid

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SECTIONS_DIR = ROOT / "presentation" / "sections"
TARGET = ROOT / "concept_transformer_presentation.ipynb"


def main(force=False):
    fragments = sorted(SECTIONS_DIR.glob("[0-9][0-9]_*.ipynb"))
    if not fragments:
        sys.exit(f"No fragments in {SECTIONS_DIR}")
    if TARGET.exists() and not force:
        existing = nbformat.read(str(TARGET), as_version=4)
        if any(cell.get("outputs") for cell in existing.cells if cell.cell_type == "code"):
            sys.exit(f"{TARGET.name} holds executed outputs; edit it directly or rerun with --force.")

    merged = nbformat.read(str(fragments[0]), as_version=4)
    seen = {cell.id for cell in merged.cells}
    for path in fragments[1:]:
        for cell in nbformat.read(str(path), as_version=4).cells:
            if cell.id in seen:
                cell.id = uuid.uuid4().hex[:8]
            seen.add(cell.id)
            merged.cells.append(cell)

    nbformat.validate(merged)
    nbformat.write(merged, str(TARGET))
    names = ", ".join(path.stem for path in fragments)
    print(f"Wrote {TARGET.name}: {len(merged.cells)} cells from {names}")


if __name__ == "__main__":
    main(force="--force" in sys.argv[1:])

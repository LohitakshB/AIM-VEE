"""Remove duplicate geometries between QM9 and QUEST structures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
except ImportError as exc:  # pragma: no cover - runtime dependency
    Chem = None
    rdDetermineBonds = None
    _RDKit_IMPORT_ERROR = exc
else:
    _RDKit_IMPORT_ERROR = None


QM9_DIR = Path("data/informed_learner/QM9_xyz_files")
QUEST_DIR = Path("data/benchmarks/QUEST_db/structures")
OUTPUT_DIR = Path("data/informed_learner/geometries_qm9_deduped_vs_quest")


def _iter_xyz_files(root: Path) -> Iterable[Path]:
    """Yield XYZ files under the provided root."""
    if not root.exists():
        return []
    return (path for path in root.rglob("*.xyz") if path.is_file())


def _xyz_canonical_smiles(path: Path) -> str:
    """Return canonical SMILES for an XYZ geometry file."""
    if Chem is None or rdDetermineBonds is None:
        raise ImportError(
            "RDKit is required for SMILES-based deduplication. "
            "Install RDKit and retry."
        ) from _RDKit_IMPORT_ERROR

    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")

    atom_count = None
    try:
        atom_count = int(lines[0].split()[0])
    except (ValueError, IndexError):
        atom_count = None

    atom_lines = lines[2:] if len(lines) > 2 else []
    xyz_lines = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        try:
            coords = [float(value) for value in parts[1:4]]
        except ValueError:
            continue
        xyz_lines.append(f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}")
        if atom_count is not None and len(xyz_lines) >= atom_count:
            break

    if not xyz_lines:
        raise ValueError(f"No atom records found in {path}")

    xyz_block = f"{len(xyz_lines)}\n{path.name}\n" + "\n".join(xyz_lines) + "\n"
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError(f"RDKit could not parse XYZ file: {path}")
    try:
        rdDetermineBonds.DetermineBonds(mol)
    except ValueError:
        rdDetermineBonds.DetermineConnectivity(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    if not smiles:
        raise ValueError(f"RDKit could not generate SMILES for: {path}")
    return smiles


def _collect_smiles(files: Iterable[Path]) -> Dict[str, Path]:
    """Map canonical SMILES to one representative path."""
    signatures: Dict[str, Path] = {}
    for path in files:
        smiles = _xyz_canonical_smiles(path)
        signatures.setdefault(smiles, path)
    return signatures


def dedupe_qm9_against_quest(
    qm9_dir: Path = QM9_DIR,
    quest_dir: Path = QUEST_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Tuple[int, int]:
    """Copy QM9 files that are not duplicates of QUEST by SMILES."""
    quest_signatures = _collect_smiles(_iter_xyz_files(quest_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for qm9_file in _iter_xyz_files(qm9_dir):
        signature = _xyz_canonical_smiles(qm9_file)
        if signature in quest_signatures:
            skipped += 1
            quest_file = quest_signatures[signature]
            print(
                f"Duplicate geometry skipped: QM9={qm9_file} QUEST={quest_file}",
                file=sys.stderr,
            )
            continue
        target = output_dir / qm9_file.name
        target.write_bytes(qm9_file.read_bytes())
        kept += 1

    return kept, skipped


if __name__ == "__main__":
    kept_count, skipped_count = dedupe_qm9_against_quest()
    print(
        "QM9 dedupe complete.",
        f"Kept {kept_count} geometries; skipped {skipped_count} duplicates.",
        f"Output: {OUTPUT_DIR}",
        sep=os.linesep,
    )

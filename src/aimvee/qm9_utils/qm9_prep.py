"""Prepare QM9-style geometry data with Bemis-Murcko scaffold splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - runtime dependency
    Chem = None
    rdDetermineBonds = None
    MurckoScaffold = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None


@dataclass(frozen=True)
class Qm9Record:
    geometry_id: str
    xyz_path: Path
    smiles: str
    scaffold: str
    target: float


def _iter_xyz_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*.xyz") if path.is_file())


def _ensure_rdkit() -> None:
    if Chem is None or rdDetermineBonds is None or MurckoScaffold is None:
        raise ImportError(
            "RDKit is required for SMILES/scaffold processing. "
            "Install RDKit and retry."
        ) from _RDKIT_IMPORT_ERROR


def _xyz_to_smiles(path: Path) -> str:
    """Return canonical SMILES from an XYZ geometry file."""
    _ensure_rdkit()

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


def _smiles_to_scaffold(smiles: str) -> str:
    _ensure_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or smiles


def _load_metadata(
    metadata_path: Path,
    id_column: str,
    target_column: str,
    smiles_column: Optional[str],
) -> Dict[str, Dict[str, str]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if metadata_path.suffix.lower() in {".json", ".jsonl"}:
        records: List[Dict[str, str]] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            if metadata_path.suffix.lower() == ".jsonl":
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            else:
                payload = json.load(handle)
                if isinstance(payload, list):
                    records = payload
                else:
                    records = list(payload.values())
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = [row for row in reader]

    mapping: Dict[str, Dict[str, str]] = {}
    for row in records:
        geometry_id = row.get(id_column, "").strip()
        if not geometry_id:
            continue
        target = row.get(target_column, "").strip()
        if target == "":
            continue
        entry = {"target": target}
        if smiles_column:
            smiles = row.get(smiles_column, "").strip()
            if smiles:
                entry["smiles"] = smiles
        mapping[geometry_id] = entry
    if not mapping:
        raise ValueError(
            f"No metadata rows matched columns id={id_column!r} target={target_column!r}."
        )
    return mapping


def _scaffold_split(
    records: Sequence[Qm9Record],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, str]:
    total = len(records)
    if total == 0:
        return {}

    scaffold_to_ids: Dict[str, List[str]] = {}
    for record in records:
        scaffold_to_ids.setdefault(record.scaffold, []).append(record.geometry_id)

    rng = random.Random(seed)
    grouped: List[Tuple[str, List[str]]] = list(scaffold_to_ids.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: len(item[1]), reverse=True)

    train_target = int(round(train_frac * total))
    val_target = int(round(val_frac * total))

    split_map: Dict[str, str] = {}
    train_count = 0
    val_count = 0

    for scaffold, ids in grouped:
        if train_count + len(ids) <= train_target:
            split = "train"
            train_count += len(ids)
        elif val_count + len(ids) <= val_target:
            split = "val"
            val_count += len(ids)
        else:
            split = "test"
        for geometry_id in ids:
            split_map[geometry_id] = split

    return split_map


def prepare_qm9_dataset(
    xyz_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    id_column: str = "id",
    target_column: str = "lowest_excited_state",
    smiles_column: Optional[str] = "smiles",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> List[Qm9Record]:
    metadata = _load_metadata(metadata_path, id_column, target_column, smiles_column)

    records: List[Qm9Record] = []
    for xyz_path in _iter_xyz_files(xyz_dir):
        geometry_id = xyz_path.stem
        info = metadata.get(geometry_id)
        if info is None:
            continue
        smiles = info.get("smiles")
        scaffold = None
        if smiles:
            try:
                scaffold = _smiles_to_scaffold(smiles)
            except ValueError:
                # Fall back to XYZ-derived SMILES if metadata SMILES are invalid.
                smiles = None
        if not smiles:
            try:
                smiles = _xyz_to_smiles(xyz_path)
                scaffold = _smiles_to_scaffold(smiles)
            except ValueError as exc:
                print(
                    f"Skipping {geometry_id}: unable to build SMILES/scaffold ({exc})."
                )
                continue
        target = float(info["target"])
        records.append(
            Qm9Record(
                geometry_id=geometry_id,
                xyz_path=xyz_path,
                smiles=smiles,
                scaffold=scaffold,
                target=target,
            )
        )

    if not records:
        raise ValueError(f"No XYZ files matched metadata in {xyz_dir}")

    split_map = _scaffold_split(records, train_frac=train_frac, val_frac=val_frac, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "qm9_metadata.csv"
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["geometry_id", "xyz_path", "smiles", "scaffold", "target", "split"])
        for record in records:
            split = split_map.get(record.geometry_id, "test")
            writer.writerow(
                [
                    record.geometry_id,
                    str(record.xyz_path),
                    record.smiles,
                    record.scaffold,
                    f"{record.target:.10f}",
                    split,
                ]
            )

    for split_name in ("train", "val", "test"):
        split_path = output_dir / f"{split_name}.csv"
        with split_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["geometry_id", "xyz_path", "smiles", "scaffold", "target"])
            for record in records:
                if split_map.get(record.geometry_id, "test") != split_name:
                    continue
                writer.writerow(
                    [
                        record.geometry_id,
                        str(record.xyz_path),
                        record.smiles,
                        record.scaffold,
                        f"{record.target:.10f}",
                    ]
                )

    split_index_path = output_dir / "splits.json"
    split_index: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for record in records:
        split_index[split_map.get(record.geometry_id, "test")].append(record.geometry_id)
    with split_index_path.open("w", encoding="utf-8") as handle:
        json.dump(split_index, handle, indent=2, sort_keys=True)

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qm9_prep",
        description="Prepare QM9-like XYZ data with Bemis-Murcko scaffold splits.",
    )
    parser.add_argument("--xyz-dir", required=True, help="Directory containing .xyz files.")
    parser.add_argument(
        "--metadata",
        required=True,
        help="CSV/JSON/JSONL file with geometry id, target, and optional smiles.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for splits.")
    parser.add_argument("--id-column", default="id", help="Metadata column for geometry id.")
    parser.add_argument(
        "--target-column",
        default="lowest_excited_state",
        help="Metadata column for lowest excited state value.",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Metadata column for SMILES (optional if XYZ -> SMILES).",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prepare_qm9_dataset(
        xyz_dir=Path(args.xyz_dir),
        metadata_path=Path(args.metadata),
        output_dir=Path(args.output_dir),
        id_column=args.id_column,
        target_column=args.target_column,
        smiles_column=args.smiles_column if args.smiles_column else None,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

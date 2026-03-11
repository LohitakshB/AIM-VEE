"""Prepare geometry data with configurable train/val/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors, rdDetermineBonds
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - runtime dependency
    Chem = None
    RDLogger = None
    rdMolDescriptors = None
    rdDetermineBonds = None
    MurckoScaffold = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None

if RDLogger is not None:
    RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class GeometryRecord:
    geometry_id: str
    xyz_path: Path
    smiles: str
    scaffold: str
    target: float
    atom_count: int


def _iter_xyz_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*.xyz") if path.is_file())


def _ensure_rdkit() -> None:
    if (
        Chem is None
        or rdDetermineBonds is None
        or MurckoScaffold is None
        or rdMolDescriptors is None
    ):
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


def _smiles_to_heavy_atom_count(smiles: str) -> int:
    _ensure_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return int(rdMolDescriptors.CalcNumHeavyAtoms(mol))


def _read_atom_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        try:
            return int(first_line.split()[0])
        except (ValueError, IndexError):
            handle.seek(0)
            atom_lines = 0
            for line in handle:
                parts = line.split()
                if len(parts) >= 4:
                    atom_lines += 1
            if atom_lines == 0:
                raise ValueError(f"Could not determine atom count for {path}")
            return atom_lines


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
        xyz_path = row.get("xyz_path", "").strip()
        geometry_path = Path(geometry_id)
        if not xyz_path and (
            geometry_path.suffix.lower() == ".xyz" or geometry_path.parent != Path(".")
        ):
            xyz_path = geometry_id
            geometry_id = geometry_path.stem
        if xyz_path:
            entry["xyz_path"] = xyz_path
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


def _split_sizes(total: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    n_train = int(round(train_frac * total))
    n_val = int(round(val_frac * total))
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def _scaffold_split(
    records: Sequence[GeometryRecord],
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


def _random_split(
    records: Sequence[GeometryRecord],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, str]:
    total = len(records)
    if total == 0:
        return {}
    rng = random.Random(seed)
    ids = [record.geometry_id for record in records]
    rng.shuffle(ids)
    n_train, n_val, _ = _split_sizes(total, train_frac, val_frac)
    split_map: Dict[str, str] = {}
    for idx, geometry_id in enumerate(ids):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_map[geometry_id] = split
    return split_map


def _extreme_value_split(
    records: Sequence[GeometryRecord],
    values: Sequence[float],
    train_frac: float,
    val_frac: float,
) -> Dict[str, str]:
    total = len(records)
    if total == 0:
        return {}
    if len(values) != total:
        raise ValueError("Extreme split values must align with records.")

    n_train, n_val, n_test = _split_sizes(total, train_frac, val_frac)
    n_tail = max(1, int(total * 0.05))
    if 4 * n_tail > total:
        n_tail = max(1, total // 4)

    sorted_pairs = sorted(zip(values, records), key=lambda item: item[0])
    ordered_records = [record for _, record in sorted_pairs]

    test_low = ordered_records[:n_tail]
    test_high = ordered_records[-n_tail:]
    val_low = ordered_records[n_tail : n_tail * 2]
    val_high = ordered_records[-n_tail * 2 : -n_tail]
    test_records = test_low + test_high
    val_records = val_low + val_high
    train_records = ordered_records[n_tail * 2 : -n_tail * 2]

    if len(train_records) != n_train or len(val_records) != n_val or len(test_records) != n_test:
        all_records = train_records + val_records + test_records
        if len(all_records) != total:
            raise ValueError("Extreme split produced inconsistent split sizes.")
        train_records = all_records[:n_train]
        val_records = all_records[n_train : n_train + n_val]
        test_records = all_records[n_train + n_val :]

    split_map: Dict[str, str] = {}
    for record in train_records:
        split_map[record.geometry_id] = "train"
    for record in val_records:
        split_map[record.geometry_id] = "val"
    for record in test_records:
        split_map[record.geometry_id] = "test"
    return split_map


def prepare_dataset(
    xyz_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    id_column: str = "id",
    target_column: str = "lowest_excited_state",
    smiles_column: Optional[str] = "smiles",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
    split_method: str = "bemis-murcko",
) -> List[GeometryRecord]:
    metadata = _load_metadata(metadata_path, id_column, target_column, smiles_column)

    records: List[GeometryRecord] = []
    skip_print_limit = 50
    skipped_prints = 0
    skipped_total = 0
    use_metadata_paths = any("xyz_path" in info for info in metadata.values())
    if use_metadata_paths:
        xyz_entries = []
        for geometry_id, info in metadata.items():
            xyz_path_str = info.get("xyz_path")
            if xyz_path_str:
                xyz_path = Path(xyz_path_str)
                if not xyz_path.is_absolute():
                    xyz_path = metadata_path.parent / xyz_path
            else:
                xyz_path = xyz_dir / f"{geometry_id}.xyz"
            xyz_entries.append((geometry_id, xyz_path, info))
    else:
        xyz_entries = []
        for xyz_path in _iter_xyz_files(xyz_dir):
            geometry_id = xyz_path.stem
            info = metadata.get(geometry_id)
            if info is None:
                continue
            xyz_entries.append((geometry_id, xyz_path, info))

    for geometry_id, xyz_path, info in xyz_entries:
        if not xyz_path.exists():
            skipped_total += 1
            if skipped_prints < skip_print_limit:
                print(f"Skipping {geometry_id}: missing XYZ file at {xyz_path}.")
                skipped_prints += 1
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
                skipped_total += 1
                if skipped_prints < skip_print_limit:
                    print(
                        f"Skipping {geometry_id}: unable to build SMILES/scaffold ({exc})."
                    )
                    skipped_prints += 1
                continue
        target = float(info["target"])
        atom_count = _read_atom_count(xyz_path)
        records.append(
            GeometryRecord(
                geometry_id=geometry_id,
                xyz_path=xyz_path,
                smiles=smiles,
                scaffold=scaffold,
                target=target,
                atom_count=atom_count,
            )
        )

    if not records:
        raise ValueError(f"No XYZ files matched metadata in {xyz_dir}")
    if skipped_total > skip_print_limit:
        print(
            "Skipping additional geometries due to SMILES/scaffold errors: "
            f"{skipped_total - skip_print_limit} more."
        )

    split_method = split_method.lower().replace("_", "-")
    if split_method == "bemis-murcko":
        split_map = _scaffold_split(
            records, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
    elif split_method == "random":
        split_map = _random_split(records, train_frac=train_frac, val_frac=val_frac, seed=seed)
    elif split_method == "size":
        split_map = _extreme_value_split(
            records,
            values=[_smiles_to_heavy_atom_count(record.smiles) for record in records],
            train_frac=train_frac,
            val_frac=val_frac,
        )
    elif split_method == "tail-split":
        split_map = _extreme_value_split(
            records,
            values=[record.target for record in records],
            train_frac=train_frac,
            val_frac=val_frac,
        )
    else:
        raise ValueError(
            "Unsupported split method. Choose from "
            "bemis-murcko, random, size, tail-split."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "metadata.csv"
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


def iter_split_csvs(
    *,
    xyz_dir: Path,
    metadata_path: Path,
    output_root: Path,
    id_column: str,
    target_column: str,
    smiles_column: Optional[str],
    train_frac: float,
    val_frac: float,
    seed: int,
    split_method: Optional[str],
    train_all_splits: bool,
    split_name: Optional[str] = None,
    predefined_train_csv: Optional[Path] = None,
    predefined_val_csv: Optional[Path] = None,
    predefined_test_csv: Optional[Path] = None,
) -> Iterable[Tuple[str, Path, Path, Path]]:
    """Generate splits and yield (split_method, train_csv, val_csv, output_dir)."""
    split_methods = (
        "random",
        "bemis-murcko",
        "size",
        "tail-split",
    )
    if train_all_splits:
        if split_name:
            raise ValueError("Use --split-name with --single-split only.")
        if split_method:
            raise ValueError("Use --train-all-splits without --split-method.")
        methods_to_run = split_methods
    else:
        methods_to_run = (split_method or "random",)

    for method in methods_to_run:
        method_norm = method.lower().replace("_", "-")
        split_label = split_name or (
            "predefined" if method_norm in ("predefined", "pre-defined") else method
        )
        if method_norm in ("predefined", "pre-defined"):
            if not predefined_train_csv or not predefined_val_csv:
                raise ValueError(
                    "Predefined splits require --predefined-train and --predefined-val."
                )
            train_csv = Path(predefined_train_csv)
            val_csv = Path(predefined_val_csv)
            if not train_csv.exists():
                raise FileNotFoundError(f"Predefined train CSV not found: {train_csv}")
            if not val_csv.exists():
                raise FileNotFoundError(f"Predefined val CSV not found: {val_csv}")
            if predefined_test_csv:
                test_csv = Path(predefined_test_csv)
                if not test_csv.exists():
                    raise FileNotFoundError(
                        f"Predefined test CSV not found: {test_csv}"
                    )
            output_dir = output_root / split_label
            output_dir.mkdir(parents=True, exist_ok=True)
            yield (split_label, train_csv, val_csv, output_dir)
            continue
        output_dir = output_root / split_label
        prepare_dataset(
            xyz_dir=xyz_dir,
            metadata_path=metadata_path,
            output_dir=output_dir,
            id_column=id_column,
            target_column=target_column,
            smiles_column=smiles_column,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
            split_method=method,
        )
        yield (
            split_label,
            output_dir / "train.csv",
            output_dir / "val.csv",
            output_dir,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_prep",
        description="Prepare XYZ data with configurable train/val/test split strategies.",
    )
    add_split_args(parser, require_paths=True)
    parser.add_argument("--output-dir", required=True, help="Output directory for splits.")
    return parser


def add_split_args(
    parser: argparse.ArgumentParser,
    *,
    require_paths: bool = False,
    default_split_method: Optional[str] = "bemis-murcko",
) -> None:
    """Add split/metadata arguments to an existing parser."""
    path_required = "required" if require_paths else "optional"
    parser.add_argument(
        "--xyz-dir",
        required=True,
        help="Directory containing .xyz files.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="CSV/JSON/JSONL file with geometry id, target, and optional smiles.",
    )
    parser.add_argument("--splits-output", required=True, help="Output directory for splits.")
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
    parser.add_argument(
        "--split-method",
        default=default_split_method,
        choices=(
            "bemis-murcko",
            "bemis_murcko",
            "random",
            "size",
            "tail-split",
            "predefined",
            "pre-defined",
        ),
        help="Split strategy for train/val/test generation.",
    )
    parser.add_argument(
        "--split-name",
        help="Override the output subdirectory name for the split (default: split method).",
    )
    split_mode = parser.add_mutually_exclusive_group()
    split_mode.add_argument(
        "--train-all-splits",
        action="store_true",
        default=True,
        help="Train sequentially on all supported split methods (default).",
    )
    split_mode.add_argument(
        "--single-split",
        dest="train_all_splits",
        action="store_false",
        help="Train only on the selected --split-method.",
    )
    parser.add_argument(
        "--predefined-train",
        help="Path to a pre-defined train CSV (required for --split-method predefined).",
    )
    parser.add_argument(
        "--predefined-val",
        help="Path to a pre-defined val CSV (required for --split-method predefined).",
    )
    parser.add_argument(
        "--predefined-test",
        help="Optional path to a pre-defined test CSV.",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prepare_dataset(
        xyz_dir=Path(args.xyz_dir),
        metadata_path=Path(args.metadata),
        output_dir=Path(args.output_dir),
        id_column=args.id_column,
        target_column=args.target_column,
        smiles_column=args.smiles_column if args.smiles_column else None,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        split_method=args.split_method,
    )


if __name__ == "__main__":
    main()

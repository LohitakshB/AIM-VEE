"""Geometry datasets and XYZ parsing utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

try:
    import periodictable as pt
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "periodictable is required to map element symbols to atomic numbers. "
        "Install it or use numeric atomic numbers in your XYZ files."
    ) from exc


def _symbol_to_atomic_number(symbol: str) -> int:
    if symbol.isdigit():
        atomic_number = int(symbol)
    else:
        try:
            atomic_number = pt.elements.symbol(symbol).number
        except Exception as exc:
            raise ValueError(f"Unsupported atom symbol: {symbol}") from exc
    if not 1 <= atomic_number <= 100:
        raise ValueError(
            f"Atomic number {atomic_number} is out of range for SchNet embeddings."
        )
    return atomic_number


def read_xyz(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        atom_count = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid XYZ header in {path}") from exc

    atom_lines = lines[2:] if len(lines) > 2 else []
    numbers: List[int] = []
    coords: List[List[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        try:
            x, y, z = (float(value) for value in parts[1:4])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate in {path}: {line}") from exc
        numbers.append(_symbol_to_atomic_number(symbol))
        coords.append([x, y, z])
        if len(numbers) >= atom_count:
            break
    if len(numbers) != atom_count:
        if len(numbers) == 0:
            raise ValueError(f"No atom records parsed in {path}")

    return torch.tensor(numbers, dtype=torch.long), torch.tensor(coords, dtype=torch.float)


def _find_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def _resolve_xyz_path(xyz_path: str, csv_path: Path) -> Path:
    path = Path(xyz_path)
    if path.is_absolute():
        return path
    candidates = [csv_path.parent / path]
    repo_root = _find_repo_root(csv_path)
    if repo_root:
        candidates.append(repo_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


class GeometryCsvDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.rows: List[Tuple[Path, float]] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                xyz_path = row.get("xyz_path", "").strip()
                target = row.get("target", "").strip()
                if not xyz_path or target == "":
                    continue
                resolved = _resolve_xyz_path(xyz_path, csv_path)
                self.rows.append((resolved, float(target)))
        if not self.rows:
            raise ValueError(f"No usable rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Data:
        xyz_path, target = self.rows[idx]
        numbers, coords = read_xyz(xyz_path)
        return Data(
            z=numbers,
            pos=coords,
            y=torch.tensor([target], dtype=torch.float),
        )


class GeometryQemfiDataset(Dataset):
    def __init__(self, rows: List[Tuple[Path, float]], qemfi_feats):
        if len(rows) != qemfi_feats.shape[0]:
            raise ValueError("Rows and QeMFi feature count mismatch.")
        self.rows = rows
        self.qemfi_feats = qemfi_feats.astype("float32", copy=False)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Data:
        xyz_path, target = self.rows[idx]
        numbers, coords = read_xyz(xyz_path)
        qemfi = torch.tensor(self.qemfi_feats[idx], dtype=torch.float32).unsqueeze(0)
        return Data(
            z=numbers,
            pos=coords,
            y=torch.tensor([target], dtype=torch.float32),
            qemfi=qemfi,
            idx=torch.tensor(idx, dtype=torch.long),
        )

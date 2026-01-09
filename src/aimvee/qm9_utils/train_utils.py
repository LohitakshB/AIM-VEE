"""Shared training utilities for QM9 baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from aimvee.utils import select_device

try:
    import periodictable as pt
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "periodictable is required to map element symbols to atomic numbers. "
        "Install it or use numeric atomic numbers in your XYZ files."
    ) from exc

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    DataStructs = None
    AllChem = None


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


class Qm9CsvDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.rows: List[Tuple[Path, float]] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                xyz_path = row.get("xyz_path", "").strip()
                target = row.get("target", "").strip()
                if not xyz_path or target == "":
                    continue
                self.rows.append((Path(xyz_path), float(target)))
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


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.z, batch.pos, batch.batch).view(-1)
        loss = torch.nn.functional.l1_loss(preds, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def eval_epoch(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.z, batch.pos, batch.batch).view(-1)
            loss = torch.nn.functional.l1_loss(preds, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)




def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def maybe_import_xyz_to_smiles() -> callable | None:
    repo_root = _repo_root()
    src_root = repo_root / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))
    try:
        from aimvee.qm9_utils.qm9_prep import _xyz_to_smiles  # noqa: SLF001
    except Exception:
        return None
    return _xyz_to_smiles


def iter_smiles_and_targets(csv_path: Path) -> Iterable[Tuple[str, float]]:
    xyz_to_smiles = maybe_import_xyz_to_smiles()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row.get("target", "").strip()
            if target == "":
                continue
            smiles = row.get("smiles", "").strip()
            if not smiles:
                xyz_path = row.get("xyz_path", "").strip()
                if not xyz_path:
                    continue
                if xyz_to_smiles is None:
                    raise ImportError(
                        "No SMILES column found and xyz-to-smiles helper is unavailable."
                    )
                smiles = xyz_to_smiles(Path(xyz_path))
            yield smiles, float(target)


def smiles_to_morgan_features(
    smiles_list: List[str], radius: int, n_bits: int
) -> np.ndarray:
    if Chem is None or DataStructs is None or AllChem is None:
        raise ImportError(
            "RDKit is required for Morgan fingerprints. Install RDKit and retry."
        )
    features = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, features[idx])
    return features


def load_morgan_dataset(
    csv_path: Path, radius: int, n_bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    smiles_list: List[str] = []
    targets: List[float] = []
    for smiles, target in iter_smiles_and_targets(csv_path):
        smiles_list.append(smiles)
        targets.append(target)
    if not smiles_list:
        raise ValueError(f"No usable rows found in {csv_path}")
    features = smiles_to_morgan_features(smiles_list, radius, n_bits)
    return features, np.asarray(targets, dtype=np.float32)


def parse_max_features(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def build_schnet_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    default_splits = repo_root / "data" / "QM9GWBSE" / "qm9_splits"
    default_output = repo_root / "models" / "schnet"

    parser = argparse.ArgumentParser(description="Train SchNet baseline on splits.")
    parser.add_argument("--train-csv", default=str(default_splits / "train.csv"))
    parser.add_argument("--val-csv", default=str(default_splits / "val.csv"))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
    parser.add_argument("--readout", choices=("add", "mean"), default="add")
    parser.add_argument("--dipole", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser




def build_rf_morgan_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    default_splits = repo_root / "data" / "QM9GWBSE" / "qm9_splits"
    default_output = repo_root / "models" / "rf_morgan"

    parser = argparse.ArgumentParser(
        description="Train a random forest regressor on Morgan fingerprints."
    )
    parser.add_argument("--train-csv", default=str(default_splits / "train.csv"))
    parser.add_argument("--val-csv", default=str(default_splits / "val.csv"))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", default="1.0")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def dump_model(model, output_path: Path) -> None:
    try:
        import joblib  # type: ignore
    except Exception:
        import pickle

        with output_path.open("wb") as handle:
            pickle.dump(model, handle)
    else:
        joblib.dump(model, output_path)

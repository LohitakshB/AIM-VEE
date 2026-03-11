"""Morgan fingerprint feature utilities."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    DataStructs = None
    AllChem = None


def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def _maybe_import_xyz_to_smiles() -> callable | None:
    repo_root = _repo_root()
    src_root = repo_root / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))
    try:
        from aimvee.data_utils.data_prep import _xyz_to_smiles  # noqa: SLF001
    except Exception:
        return None
    return _xyz_to_smiles


def iter_smiles_and_targets(csv_path: Path) -> Iterable[Tuple[str, float]]:
    xyz_to_smiles = _maybe_import_xyz_to_smiles()
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

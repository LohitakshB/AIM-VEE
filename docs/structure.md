# Codebase structure

This repo is organized around reusable modules and CLI-driven experiments.

- `src/aimvee/`: core library package.
- `src/aimvee/cli.py`: CLI entrypoints (`aimvee <command>`).
- `src/aimvee/api.py`: Python API wrappers for the main workflows.
- `src/aimvee/datasets`: dataset wrappers and XYZ parsing.
- `src/aimvee/features`: feature generators (Coulomb matrices, Morgan fingerprints).
- `src/aimvee/models`: model definitions (QeMFi surrogate, MFF-MLP).
- `src/aimvee/trainers`: shared training loops (PyTorch/Torch Geometric).
- `src/aimvee/experiments`: experiment runners used by the CLI.
- `scripts/generate_metadata.py`: standalone helper for creating QM9 metadata CSVs.
- `docs/`: documentation and workflows.
- `notebooks/`: exploratory analysis and prototyping.
- `data/`, `models/`, `outputs/`: datasets and generated artifacts (ignored by git).

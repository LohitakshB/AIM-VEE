"""Command-line entrypoints for AIM-VEE workflows."""

from __future__ import annotations

import argparse

from aimvee.data_utils.data_prep import add_split_args
from aimvee.experiments.chemprop import build_parser as build_chemprop_parser
from aimvee.experiments.chemprop import run_chemprop
from aimvee.experiments.data_prep import build_parser as build_data_prep_parser
from aimvee.experiments.data_prep import run_data_prep
from aimvee.experiments.evaluate import build_parser as build_eval_parser
from aimvee.experiments.evaluate import run_evaluation
from aimvee.experiments.evaluate_model import (
    build_parser as build_eval_model_parser,
    run_evaluate_model,
)
from aimvee.experiments.mff_mlp import build_parser as build_mff_mlp_parser
from aimvee.experiments.mff_mlp import run_mff_mlp
from aimvee.experiments.umff_mlp import build_parser as build_umff_mlp_parser
from aimvee.experiments.umff_mlp import run_umff_mlp
from aimvee.experiments.qemfi import (
    run_generate_cm,
    run_prep_qemfi,
    run_train_qemfi,
)
from aimvee.experiments.rf_morgan import build_parser as build_rf_morgan_parser
from aimvee.experiments.rf_morgan import run_rf_morgan
from aimvee.experiments.schnet import build_parser as build_schnet_parser
from aimvee.experiments.schnet import run_schnet


def _add_qemfi_parsers(subparsers: argparse._SubParsersAction) -> None:
    cm = subparsers.add_parser(
        "generate-cm", help="Generate Coulomb matrices from QeMFi npz files."
    )
    cm.add_argument("--input-dir", required=True)
    cm.add_argument("--output-dir", required=True)
    cm.set_defaults(func=run_generate_cm)

    prep = subparsers.add_parser("prep-qemfi", help="Prepare QeMFi datasets.")
    prep.add_argument("--qemfi-dir", required=True)
    prep.add_argument("--reps-dir", required=True)
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--method", default="CM")
    prep.add_argument(
        "--molecules",
        default="urea,acrolein,alanine,sma,nitrophenol,urocanic,dmabn,thymine,o-hbdi",
    )
    prep.add_argument("--n-geom-per-mol", type=int, default=15000)
    prep.set_defaults(func=run_prep_qemfi)

    train = subparsers.add_parser("train-qemfi", help="Train QeMFi surrogate.")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--method", default="CM")
    train.add_argument("--pca-components", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--emb-dim", type=int, default=32)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--device", default="")
    train.set_defaults(func=run_train_qemfi)


def _add_split_experiment(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
    base_parser: argparse.ArgumentParser,
    run_func,
) -> None:
    parser = subparsers.add_parser(name, parents=[base_parser], help=help_text)
    add_split_args(parser, default_split_method=None)
    parser.set_defaults(id_column="geometry", target_column="output")
    parser.set_defaults(func=run_func)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimvee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_qemfi_parsers(subparsers)

    data_prep = subparsers.add_parser(
        "prepare-splits",
        parents=[build_data_prep_parser(add_help=False)],
        help="Prepare split CSVs.",
    )
    data_prep.set_defaults(func=run_data_prep)

    _add_split_experiment(
        subparsers,
        name="train-schnet",
        help_text="Train SchNet baseline.",
        base_parser=build_schnet_parser(add_help=False),
        run_func=run_schnet,
    )
    _add_split_experiment(
        subparsers,
        name="train-chemprop",
        help_text="Train Chemprop baseline.",
        base_parser=build_chemprop_parser(add_help=False),
        run_func=run_chemprop,
    )
    _add_split_experiment(
        subparsers,
        name="train-rf-morgan",
        help_text="Train RF Morgan baseline.",
        base_parser=build_rf_morgan_parser(add_help=False),
        run_func=run_rf_morgan,
    )

    mff_parser = subparsers.add_parser(
        "train-mff-mlp",
        parents=[build_mff_mlp_parser(add_help=False)],
        help="Train MFF-MLP on QeMFi + SchNet features.",
    )
    mff_parser.set_defaults(func=run_mff_mlp)

    umff_parser = subparsers.add_parser(
        "train-umff-mlp",
        parents=[build_umff_mlp_parser(add_help=False)],
        help="Train UMFF-MLP with uncertainty quantification.",
    )
    umff_parser.set_defaults(func=run_umff_mlp)

    eval_parser = subparsers.add_parser(
        "evaluate",
        parents=[build_eval_parser(add_help=False)],
        help="Evaluate model predictions and compare methods.",
    )
    eval_parser.set_defaults(func=run_evaluation)

    eval_model_parser = subparsers.add_parser(
        "evaluate-model",
        parents=[build_eval_model_parser(add_help=False)],
        help="Run model inference on an input CSV and generate graphs.",
    )
    eval_model_parser.set_defaults(func=run_evaluate_model)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

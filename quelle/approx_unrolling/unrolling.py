import argparse
import os

from quelle.approx_unrolling.EK_FAC import compute_EK_FAC_checkpoints
from quelle.approx_unrolling.language_task import LanguageModelingTask
from quelle.approx_unrolling.model_checkpoints import PythiaCheckpoints
from quelle.approx_unrolling.pile_data import get_pile_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path that is storing models and (approximate) Hessians.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=False,
        help="Whether to use torch compile for computing factors and scores.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Batch size for computing matrices.",
    )
    parser.add_argument(
        "--compute_per_token_scores",
        action="store_true",
        default=False,
        help="Boolean flag to compute per token scores.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )

    parser.add_argument(
        "--pythia_model_name",
        type=str,
        default="EleutherAI/pythia-14m",
        help="Pythia model name.",
    )

    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[2000, 3000, 4000],
        help="List of checkpoints to load.",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def run():
    args = parse_args()

    checkpoint_list = [2000, 3000, 4000, 6000]
    model_name = args.pythia_model_name
    checkpoints_dir = args.checkpoint_dir

    pythia_checkpoints_manager = PythiaCheckpoints(
        checkpoint_list, model_name, checkpoints_dir
    )
    pythia_checkpoints_manager.save_models(overwrite=True)

    assert pythia_checkpoints_manager.module_keys is not None

    task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)

    train_dataset = get_pile_dataset(model_str=args.pythia_model_name, step=0)

    compute_EK_FAC_checkpoints(
        checkpoint_manager=pythia_checkpoints_manager,
        task=task,
        train_dataset=train_dataset,
        EK_FAC_args=args,
    )


if __name__ == "__main__":
    run()

    # args = SimpleNamespace(
    #     query_batch_size=32,
    #     train_batch_size=64,
    #     use_half_precision=True,
    #     per_device_batch_size=32,
    # )

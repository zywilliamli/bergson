"""Generate and cache the synthetic facts datasets used by ``bergson recall``.

The statements dataset (one row per paraphrase template of a person+field
fact) is the corpus the model is trained on and scored over; the questions
dataset (one row per distinct person+field) provides the queries. Both carry
``identifier``/``field`` metadata so scores can be mapped back to the gold
statements by row index.

Datasets can also be generated ahead of a pipeline run with e.g.::

    python -m bergson.recall.generate --num_people 1000
"""

from pathlib import Path

from datasets import Dataset
from simple_parsing import ArgumentParser

from bergson.config.config import RecallDataConfig
from bergson.recall.facts import fact_generator

NUM_FIELDS = 4
"""Fields per profile (birthdate, birthplace, employer, university)."""


def recall_dataset_paths(recall_data_cfg: RecallDataConfig) -> tuple[Path, Path]:
    """Canonical cache paths for the (statements, questions) datasets."""
    tag = f"{recall_data_cfg.num_people}p_seed{recall_data_cfg.seed}"
    if recall_data_cfg.single_paraphrase:
        tag += "_single"

    data_dir = Path(recall_data_cfg.data_dir)
    return data_dir / f"statements_{tag}.hf", data_dir / f"questions_{tag}.hf"


def ensure_recall_datasets(recall_data_cfg: RecallDataConfig) -> tuple[Path, Path]:
    """Generate + cache the recall datasets, or reuse existing ones.

    Returns the (statements, questions) dataset paths. Generation is fully
    seeded, so the same config always produces identical datasets.
    """
    statements_path, questions_path = recall_dataset_paths(recall_data_cfg)
    if statements_path.exists() and questions_path.exists():
        print(
            f"Reusing cached recall datasets at {statements_path} "
            f"and {questions_path}"
        )
        return statements_path, questions_path

    statements = []
    questions = []
    seen_questions = set()

    # fact_generator counts person+field groups, so 4 groups per person.
    rows = fact_generator(
        num_facts=NUM_FIELDS * recall_data_cfg.num_people,
        path=recall_data_cfg.data_dir,
        seed=recall_data_cfg.seed,
    )
    for row in rows:
        if recall_data_cfg.single_paraphrase and row["template"] != 0:
            continue
        statements.append(row)

        key = (row["identifier"], row["field"])
        if key not in seen_questions:
            seen_questions.add(key)
            questions.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    # Combined column for models without a chat template;
                    # use prompt/completion columns to mask loss to the
                    # answer span when the tokenizer supports chat templates.
                    "text": f"{row['question']} {row['answer']}",
                    "field": row["field"],
                    "identifier": row["identifier"],
                }
            )

    num_people = len({s["identifier"] for s in statements})
    if num_people < recall_data_cfg.num_people:
        raise ValueError(
            f"Only generated {num_people} of the requested "
            f"{recall_data_cfg.num_people} people; the name lists in "
            f"{recall_data_cfg.data_dir}/names are too small."
        )

    Dataset.from_list(statements).save_to_disk(str(statements_path))
    Dataset.from_list(questions).save_to_disk(str(questions_path))

    print(
        f"Generated {len(statements)} statements and {len(questions)} "
        f"questions for {num_people} people at {statements_path} and "
        f"{questions_path}"
    )
    print(
        "Reproduce with: python -m bergson.recall.generate "
        f"--num_people {recall_data_cfg.num_people} "
        f"--seed {recall_data_cfg.seed} "
        f"--data_dir {recall_data_cfg.data_dir}"
        + (" --single_paraphrase" if recall_data_cfg.single_paraphrase else "")
    )
    return statements_path, questions_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(RecallDataConfig, dest="recall_data_cfg")
    args = parser.parse_args()
    ensure_recall_datasets(args.recall_data_cfg)

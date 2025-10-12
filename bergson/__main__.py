from simple_parsing import parse

from .build import build_gradient_dataset
from .data import IndexConfig


def main():
    cfg = parse(IndexConfig)

    if not cfg.save_index and not cfg.save_processor:
        raise ValueError("At least one of save_index or save_processor must be True")

    build_gradient_dataset(cfg)


if __name__ == "__main__":
    main()

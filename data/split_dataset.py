"""
data/split_dataset.py
=====================
Reads resume_dataset.json and creates train.json / val.json / test.json
with a stratified-random 70 / 15 / 15 split.

Usage:
    python data/split_dataset.py
    python data/split_dataset.py --seed 99 --train 0.75 --val 0.125 --test 0.125

Output files are written to the data/ directory (same directory as this script).
"""

import argparse
import json
import random
from pathlib import Path


# ── Default split ratios ──────────────────────────────────────────────────────
DEFAULT_TRAIN = 0.70
DEFAULT_VAL   = 0.15
DEFAULT_TEST  = 0.15
DEFAULT_SEED  = 42


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}.")
    print(f"  Loaded {len(data)} examples from {path.name}")
    return data


def split(
    data: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split the dataset. Ratios must sum to 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (got {total:.4f}).")

    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val   = round(n * val_ratio)
    # test gets the remainder to avoid rounding drift
    n_test  = n - n_train - n_val

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    return train, val, test


def save(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved  {len(data):>3} examples -> {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split resume_dataset.json into train/val/test.")
    parser.add_argument("--seed",  type=int,   default=DEFAULT_SEED)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN)
    parser.add_argument("--val",   type=float, default=DEFAULT_VAL)
    parser.add_argument("--test",  type=float, default=DEFAULT_TEST)
    args = parser.parse_args()

    data_dir    = Path(__file__).parent
    source_path = data_dir / "resume_dataset.json"

    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found.\n"
            "Please create data/resume_dataset.json first (Step 4 of the plan)."
        )

    print(f"\nSplitting dataset with seed={args.seed}, "
          f"train={args.train}, val={args.val}, test={args.test}")

    data = load_dataset(source_path)

    if len(data) < 10:
        raise ValueError(
            f"Only {len(data)} examples found. Need at least 10 to split meaningfully."
        )

    train, val, test = split(data, args.train, args.val, args.test, args.seed)

    save(train, data_dir / "train.json")
    save(val,   data_dir / "val.json")
    save(test,  data_dir / "test.json")

    print(f"\nDone. Total: {len(data)} -> "
          f"train={len(train)}, val={len(val)}, test={len(test)}")
    print("NOTE: test.json must NEVER be used during training or hyperparameter tuning.")


if __name__ == "__main__":
    main()

"""
Validation splitting module: expanding rolling-origin evaluation.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_idx: int
    origin_step: int
    train_indices: np.ndarray
    val_indices: np.ndarray


def generate_rolling_origin_folds(
    n_train_points: int = 110,
    horizon: int = 24,
    min_train_len: int = 50,
    step_size: int = 8,
) -> list[Fold]:
    """
    Generates expanding-window rolling origins inside the development set (110 points).
    At origin t, training is 0..t-1, validation is t..(t + horizon - 1).
    Ensures t + horizon <= n_train_points.
    """
    folds = []
    fold_idx = 0

    origin = min_train_len
    while origin + horizon <= n_train_points:
        train_idx = np.arange(0, origin)
        val_idx = np.arange(origin, origin + horizon)

        folds.append(
            Fold(
                fold_idx=fold_idx,
                origin_step=origin,
                train_indices=train_idx,
                val_indices=val_idx,
            )
        )
        fold_idx += 1
        origin += step_size

    # Always include the latest possible origin within development set if not already aligned
    max_possible_origin = n_train_points - horizon
    if folds[-1].origin_step != max_possible_origin:
        train_idx = np.arange(0, max_possible_origin)
        val_idx = np.arange(max_possible_origin, n_train_points)
        folds.append(
            Fold(
                fold_idx=fold_idx,
                origin_step=max_possible_origin,
                train_indices=train_idx,
                val_indices=val_idx,
            )
        )

    return folds


if __name__ == "__main__":
    folds = generate_rolling_origin_folds(n_train_points=110, horizon=24)
    print(f"Generated {len(folds)} rolling-origin folds inside the 110-step training set:")
    for f in folds:
        print(
            f"Fold {f.fold_idx}: Train len = {len(f.train_indices)} (0..{f.train_indices[-1]}), "
            f"Val len = {len(f.val_indices)} (origin={f.origin_step}, steps {f.val_indices[0]}..{f.val_indices[-1]})"
        )
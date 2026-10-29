"""Train the ridge model and write submission.csv."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from helpers import create_csv_submission
from implementations import ridge_regression

ROOT = Path(__file__).resolve().parent
SEED = 42
LAMBDA = 0.001


def load_data(data_dir):
    """Read the CSV files, or unpack the supplied archive first."""
    filenames = ("x_train.csv", "y_train.csv", "x_test.csv")
    if not all((data_dir / name).is_file() for name in filenames):
        archive_path = ROOT / "data" / "dataset_to_release.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Put {', '.join(filenames)} in {data_dir}, "
                f"or put the course archive at {archive_path}."
            )
        data_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive_path) as archive:
            for name in filenames:
                if not (data_dir / name).is_file():
                    (data_dir / name).write_bytes(archive.read(f"dataset/{name}"))

    def read(name):
        # Empty fields become NaN. loadtxt uses less memory than genfromtxt here.
        return np.loadtxt(
            data_dir / name,
            delimiter=",",
            skiprows=1,
            converters=lambda value: float(value) if value else np.nan,
        )

    train = read("x_train.csv")
    labels = read("y_train.csv")
    test = read("x_test.csv")
    if not np.array_equal(train[:, 0], labels[:, 0]):
        raise ValueError("Training IDs and label IDs do not match.")
    if not np.isin(labels[:, 1], [-1, 1]).all():
        raise ValueError("Expected labels -1 and 1.")
    return train[:, 1:], test[:, 1:], labels[:, 1], test[:, 0].astype(int)


def fit_preprocessor(x):
    """Learn which columns to keep, their medians, and their scaling."""
    columns = np.flatnonzero(np.isfinite(x).any(axis=0))
    clean = x[:, columns].copy()
    clean[~np.isfinite(clean)] = np.nan
    medians = np.nanmedian(clean, axis=0)
    clean = np.where(np.isnan(clean), medians, clean)
    means = clean.mean(axis=0)
    scales = clean.std(axis=0)
    keep = scales > 1e-12
    return {
        "columns": columns[keep],
        "medians": medians[keep],
        "means": means[keep],
        "scales": scales[keep],
    }


def transform(x, stats):
    """Use the fitted statistics and add an intercept."""
    clean = x[:, stats["columns"]].copy()
    clean = np.where(np.isfinite(clean), clean, stats["medians"])
    clean = (clean - stats["means"]) / stats["scales"]
    return np.column_stack((np.ones(len(clean)), clean))


def metrics(y, predictions):
    """Classification metrics for the positive class (+1)."""
    tp = np.sum((y == 1) & (predictions == 1))
    fp = np.sum((y == -1) & (predictions == 1))
    fn = np.sum((y == 1) & (predictions == -1))
    return {
        "accuracy": float(np.mean(y == predictions)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    x, x_test, y, test_ids = load_data(args.data_dir)
    print(f"Training: {x.shape}; test: {x_test.shape}", flush=True)

    indices = np.random.default_rng(SEED).permutation(len(y))
    cut = int(0.8 * len(y))
    train_idx, valid_idx = indices[:cut], indices[cut:]

    # Fit preprocessing on the training split only.
    stats = fit_preprocessor(x[train_idx])
    tx_train = transform(x[train_idx], stats)
    tx_valid = transform(x[valid_idx], stats)
    y_train = (y[train_idx] == 1).astype(float)
    w, _ = ridge_regression(y_train, tx_train, LAMBDA)
    scores = tx_valid @ w
    del tx_train, tx_valid

    thresholds = np.linspace(0.05, 0.5, 46)
    f1_scores = []
    for t in thresholds:
        predictions = np.where(scores >= t, 1, -1)
        f1_scores.append(metrics(y[valid_idx], predictions)["f1"])
    threshold = float(thresholds[np.argmax(f1_scores)])

    results = {
        "seed": SEED,
        "lambda": LAMBDA,
        "training_rows": len(train_idx),
        "validation_rows": len(valid_idx),
        "test_rows": len(test_ids),
        "original_features": x.shape[1],
        "validation_retained_features": len(stats["columns"]),
        "positive_fraction": float(np.mean(y == 1)),
        "always_negative": metrics(y[valid_idx], -np.ones(len(valid_idx))),
        "ridge_threshold_0.5": metrics(y[valid_idx], np.where(scores >= 0.5, 1, -1)),
        "selected_threshold": threshold,
        "ridge_selected_threshold": metrics(
            y[valid_idx], np.where(scores >= threshold, 1, -1)
        ),
    }
    print(json.dumps(results, indent=2), flush=True)

    print("Refitting on all training rows...", flush=True)
    stats = fit_preprocessor(x)
    tx = transform(x, stats)
    del x
    w, _ = ridge_regression((y == 1).astype(float), tx, LAMBDA)
    del tx
    predictions = np.where(transform(x_test, stats) @ w >= threshold, 1, -1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, predictions, str(args.output))
    results["final_retained_features"] = len(stats["columns"])
    results_path = args.output.with_suffix(".metrics.json")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)
    print(f"Saved {results_path}", flush=True)


if __name__ == "__main__":
    main()

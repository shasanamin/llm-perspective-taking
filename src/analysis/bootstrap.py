from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.utils.common import slugify


DEFAULT_SAMPLE_SIZES = [1, 2, 3, 5, 10]


def mean_se(series: pd.Series) -> tuple[float, float]:
    values = series.dropna()
    if values.empty:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values.iloc[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def bootstrap_annotations(
    annotations: pd.DataFrame,
    sample_size: int = 1,
    iterations: int = 1_000,
    random_state: int = 0,
) -> pd.DataFrame:
    required_columns = {"comment_id", "value", "ground_truth"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for bootstrap: {sorted(missing_columns)}")

    frame = annotations[["comment_id", "value", "ground_truth"]].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["ground_truth"] = pd.to_numeric(frame["ground_truth"], errors="coerce")
    frame = frame.dropna(subset=["value", "ground_truth"])

    rng = np.random.default_rng(random_state)
    item_ids = sorted(frame["comment_id"].unique())
    stats = {
        "mean": np.full(len(item_ids), np.nan),
        "var": np.full(len(item_ids), np.nan),
        "ci_low": np.full(len(item_ids), np.nan),
        "ci_high": np.full(len(item_ids), np.nan),
        "mse": np.full(len(item_ids), np.nan),
    }

    ground_truth_lookup = (
        frame[["comment_id", "ground_truth"]]
        .drop_duplicates("comment_id")
        .set_index("comment_id")["ground_truth"]
    )

    for row_index, comment_id in enumerate(item_ids):
        values = frame.loc[frame["comment_id"] == comment_id, "value"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        boot_means = rng.choice(values, size=(iterations, sample_size), replace=True).mean(axis=1)
        truth = float(ground_truth_lookup.at[comment_id])

        stats["mean"][row_index] = boot_means.mean()
        stats["var"][row_index] = boot_means.var(ddof=0)
        stats["ci_low"][row_index] = np.percentile(boot_means, 5)
        stats["ci_high"][row_index] = np.percentile(boot_means, 95)
        stats["mse"][row_index] = mean_squared_error(
            np.full(iterations, truth),
            boot_means,
        )

    result = pd.DataFrame(
        {
            "comment_id": item_ids,
            "ground_truth": [float(ground_truth_lookup.at[item_id]) for item_id in item_ids],
        }
    )
    for column_name, values in stats.items():
        result[column_name] = values
    result["bias"] = result["mean"] - result["ground_truth"]
    return result


def summarize_grouped_annotations(
    annotations: pd.DataFrame,
    group_columns: list[str],
    sample_sizes: list[int] | None = None,
    iterations: int = 1_000,
    cache_dir: str | Path | None = None,
    cache_prefix: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    if sample_sizes is None:
        sample_sizes = DEFAULT_SAMPLE_SIZES

    rows: list[dict[str, Any]] = []
    cache_root = Path(cache_dir) if cache_dir is not None else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    grouped = annotations.groupby(group_columns, dropna=False)
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_data = dict(zip(group_columns, group_key))

        for sample_size in sample_sizes:
            cache_path = None
            if cache_root is not None and cache_prefix is not None:
                key_slug = "__".join(slugify(str(group_data[column])) for column in group_columns)
                filename = f"{cache_prefix}__{key_slug}__n{sample_size}.csv"
                # Truncate long filenames using a hash to stay within OS limits
                if len(filename) > 200:
                    short_hash = hashlib.sha1(key_slug.encode()).hexdigest()[:12]
                    filename = f"{cache_prefix}__{short_hash}__n{sample_size}.csv"
                cache_path = cache_root / filename

            if cache_path is not None and cache_path.exists() and not force:
                boot_df = pd.read_csv(cache_path)
            else:
                boot_df = bootstrap_annotations(
                    group_frame,
                    sample_size=sample_size,
                    iterations=iterations,
                )
                if cache_path is not None:
                    boot_df.to_csv(cache_path, index=False)

            bias_mean, bias_se = mean_se(boot_df["bias"])
            var_mean, var_se = mean_se(boot_df["var"])
            mse_mean, mse_se = mean_se(boot_df["mse"])

            row = dict(group_data)
            row.update(
                {
                    "n_annotators": sample_size,
                    "n_items": int(boot_df["comment_id"].nunique()),
                    "bias_mean": bias_mean,
                    "bias_se": bias_se,
                    "var_mean": var_mean,
                    "var_se": var_se,
                    "mse_mean": mse_mean,
                    "mse_se": mse_se,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)

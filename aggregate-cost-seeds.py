import os
import re
import glob
import argparse
import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "LOGD", "CB1", "DPP4", "HIVINT", "HIVPROT", "METAB",
    "NK1", "OX1", "OX2", "3A4", "PGP", "PPB", "RAT_F", "TDI", "THROMBIN"
]
# DEFAULT_DATASETS = [
#     "PGP", "PPB", "RAT_F", "TDI", "THROMBIN"
# ]
# Columns in one seed-level summary_results.csv.
# The first entry is the output suffix. For example,
# ("fdr", "mean_fdr", "std_fdr") becomes mean_fdr and sd_fdr
# in the aggregated_over_seeds.csv file.
METRIC_SPECS = [
    ("eta", "mean_eta", None),
    ("fdr", "mean_fdr", "std_fdr"),
    ("power", "mean_power", "std_power"),
    ("average_cost", "mean_average_cost", "std_average_cost"),
    ("n_selected", "mean_n_selected", "std_n_selected"),
    ("n_selected_fail", "mean_n_selected_fail", "std_n_selected_fail"),
    ("n_selected_ind", "mean_n_selected_ind", "std_n_selected_ind"),
    ("n_selected_pass", "mean_n_selected_pass", "std_n_selected_pass"),
    ("selected_frac_fail", "selected_frac_fail", None),
    ("selected_frac_ind", "selected_frac_ind", None),
    ("selected_frac_pass", "selected_frac_pass", None),
]

GROUP_COLS = ["method", "q", "gamma_ratio", "lambda"]


def extract_seed(file_path):
    """Extract seed value from names like '3A4 1.00 seed_17 summary_results.csv'."""
    filename = os.path.basename(file_path)
    match = re.search(r" seed_(\d+)", filename)
    if match is None:
        return np.nan
    return int(match.group(1))


def add_selected_fractions(df):
    """Add fail/ind/pass fractions among selected compounds if count columns exist."""
    required = [
        "mean_n_selected",
        "mean_n_selected_fail",
        "mean_n_selected_ind",
        "mean_n_selected_pass",
    ]
    if not all(col in df.columns for col in required):
        return df

    denom = pd.to_numeric(df["mean_n_selected"], errors="coerce")
    denom = denom.replace(0, np.nan)

    df = df.copy()
    df["selected_frac_fail"] = pd.to_numeric(df["mean_n_selected_fail"], errors="coerce") / denom
    df["selected_frac_ind"] = pd.to_numeric(df["mean_n_selected_ind"], errors="coerce") / denom
    df["selected_frac_pass"] = pd.to_numeric(df["mean_n_selected_pass"], errors="coerce") / denom
    return df


def pooled_mean_sd(group, mean_col, std_col=None, n_col="n_trials"):
    """
    Combine seed-level summary rows into one mean and one sample standard deviation.

    If each summary_results.csv row is based on n_trials=1, this is simply the
    mean and sample standard deviation over seeds. If n_trials > 1 and the
    corresponding std_* column is available, the function uses the usual pooled
    variance formula, combining within-summary and between-summary variation.
    """
    means = pd.to_numeric(group[mean_col], errors="coerce")

    if n_col in group.columns:
        n = pd.to_numeric(group[n_col], errors="coerce")
    else:
        n = pd.Series(1.0, index=group.index)

    valid = means.notna() & n.notna() & (n > 0)
    means = means[valid].astype(float)
    n = n[valid].astype(float)

    if len(means) == 0:
        return np.nan, np.nan

    total_n = n.sum()
    mean = (n * means).sum() / total_n

    if total_n <= 1:
        return mean, np.nan

    # Between-summary variation.
    between_ss = (n * (means - mean) ** 2).sum()

    # Within-summary variation, if available. If n_trials=1, this contributes 0.
    within_ss = 0.0
    if std_col is not None and std_col in group.columns:
        sds = pd.to_numeric(group.loc[valid.index[valid], std_col], errors="coerce")
        sds = sds.reindex(means.index).fillna(0.0).astype(float)
        within_ss = ((n - 1).clip(lower=0) * (sds ** 2)).sum()

    sd = np.sqrt((within_ss + between_ss) / (total_n - 1))
    return mean, sd


def aggregate_one_dataset(dataset, sample, base_dir, expected_n_files=None):
    dataset_dir = os.path.join(base_dir, f"{dataset} {sample:.2f}")
    pattern = os.path.join(
        dataset_dir,
        f"{dataset} {sample:.2f} seed_* summary_results.csv"
    )
    file_list = sorted(glob.glob(pattern))

    if len(file_list) == 0:
        print(f"[WARN] No summary files found for {dataset}: {pattern}")
        return None

    if expected_n_files is not None and len(file_list) != expected_n_files:
        print(
            f"[WARN] {dataset}: found {len(file_list)} summary files, "
            f"but expected {expected_n_files}."
        )

    df_list = []
    for file_path in file_list:
        df = pd.read_csv(file_path)
        df = add_selected_fractions(df)
        df["seed"] = extract_seed(file_path)
        df["source_file"] = os.path.basename(file_path)
        df_list.append(df)

    all_df = pd.concat(df_list, ignore_index=True)

    missing_group_cols = [col for col in GROUP_COLS if col not in all_df.columns]
    if missing_group_cols:
        raise ValueError(f"{dataset}: missing grouping columns: {missing_group_cols}")

    available_specs = [
        (suffix, mean_col, std_col)
        for suffix, mean_col, std_col in METRIC_SPECS
        if mean_col in all_df.columns
    ]

    def summarize_group(group):
        out = {
            "n_summaries": len(group),
            "n_seeds": group["seed"].nunique(dropna=True),
        }
        if "n_trials" in group.columns:
            out["total_n_trials"] = pd.to_numeric(
                group["n_trials"], errors="coerce"
            ).fillna(0).sum()
        else:
            out["total_n_trials"] = len(group)

        for suffix, mean_col, std_col in available_specs:
            mean_value, sd_value = pooled_mean_sd(group, mean_col, std_col)
            out[f"mean_{suffix}"] = mean_value
            out[f"sd_{suffix}"] = sd_value

        return pd.Series(out)

    rows = []
    for key_values, group in all_df.groupby(GROUP_COLS, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(GROUP_COLS, key_values))
        row.update(summarize_group(group).to_dict())
        rows.append(row)

    agg_df = (
        pd.DataFrame(rows)
        .sort_values(GROUP_COLS, na_position="last")
        .reset_index(drop=True)
    )

    # Put the most commonly used columns first, while keeping any dynamically
    # generated columns afterward.
    preferred_order = [
        "method", "q", "gamma_ratio", "lambda",
        "mean_eta", "sd_eta",
        "mean_fdr", "sd_fdr",
        "mean_power", "sd_power",
        "mean_average_cost", "sd_average_cost",
        "mean_n_selected", "sd_n_selected",
        "mean_n_selected_fail", "sd_n_selected_fail",
        "mean_n_selected_ind", "sd_n_selected_ind",
        "mean_n_selected_pass", "sd_n_selected_pass",
        "mean_selected_frac_fail", "sd_selected_frac_fail",
        "mean_selected_frac_ind", "sd_selected_frac_ind",
        "mean_selected_frac_pass", "sd_selected_frac_pass",
        "n_summaries", "n_seeds", "total_n_trials",
    ]
    ordered_cols = [col for col in preferred_order if col in agg_df.columns]
    other_cols = [col for col in agg_df.columns if col not in ordered_cols]
    agg_df = agg_df[ordered_cols + other_cols]

    out_file = os.path.join(
        dataset_dir,
        f"{dataset} {sample:.2f} aggregated_over_seeds.csv"
    )
    agg_df.to_csv(out_file, index=False)

    print(
        f"[OK] Saved {out_file} using {len(file_list)} summary files; "
        f"output rows = {len(agg_df)}."
    )
    return agg_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate RSI-EC/RSI-CS summary_results.csv files over seeds."
    )
    parser.add_argument("--base_dir", default="result-cost-0516")
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to aggregate. Default: all 15 datasets.",
    )
    parser.add_argument(
        "--expected_n_files",
        type=int,
        default=100,
        help="Expected number of seed summary files per dataset. Use 0 to disable the warning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    expected = None if args.expected_n_files == 0 else args.expected_n_files

    for dataset_name in args.datasets:
        aggregate_one_dataset(
            dataset=dataset_name,
            sample=args.sample,
            base_dir=args.base_dir,
            expected_n_files=expected,
        )

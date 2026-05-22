import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

SCRIPT_VERSION = "method-separated-direct-seed-mean-v1"

ALL_DATASETS = [
    "3A4", "CB1", "DPP4",
    "HIVINT", "HIVPROT", "LOGD",
    "METAB", "NK1", "OX1",
    "OX2", "PGP", "PPB",
    "RAT_F", "TDI", "THROMBIN",
]

LIGHT_GRAY = "#D9D9D9"
COST_COLOR = "#5CBF60"
POWER_COLOR = "#8C6BB1"
FDP_COLOR = "#E85B5B"
DELTA_ZERO_COLOR = "#666666"

TITLE_SIZE = 26
AXIS_LABEL_SIZE = 25
TICK_LABEL_SIZE = 23
XTICK_LABEL_SIZE = 23
PANEL_TITLE_SIZE = 36
LEGEND_FONT_SIZE = 36

MAIN_LINE_WIDTH = 3.2
TARGET_LINE_WIDTH = 2.5
MARKER_SIZE = 7.5
LEGEND_LINE_WIDTH = 5.2
LEGEND_MARKER_SIZE = 13.0

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#666666",
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "grid.color": LIGHT_GRAY,
    "grid.alpha": 0.8,
    "grid.linestyle": "-",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": LEGEND_FONT_SIZE,
})


def parse_eta_grid(grid_string):
    if grid_string is None or str(grid_string).strip().lower() in {"", "all", "none"}:
        return None

    values = []
    for item in str(grid_string).split(","):
        item = item.strip()
        if item:
            values.append(float(item))

    if not values:
        return None

    return np.array(values, dtype=float)


def q_to_tag(q):
    return f"q{int(round(q * 10)):02d}"


def method_to_tag(method):
    return method.replace("RSI-", "").lower()


def format_param_label(value):
    if pd.isna(value):
        return ""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def compact_y_tick_label(y, pos):
    if not np.isfinite(y):
        return ""
    ay = abs(y)
    if ay >= 10:
        return f"{y:.0f}"
    if ay >= 1:
        return f"{y:.1f}".rstrip("0").rstrip(".")
    if ay >= 0.1:
        return f"{y:.2f}".rstrip("0").rstrip(".")
    return f"{y:.2f}"


def nice_upper_bound(value, default=1.0):
    if value is None or not np.isfinite(value) or value <= 0:
        return default
    return 1.10 * value


def regular_axis_text(ax):
    ax.title.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")
    ax.title.set_fontsize(PANEL_TITLE_SIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(TICK_LABEL_SIZE)
        tick_label.set_fontweight("normal")


def regular_legend_text(legend):
    if legend is None:
        return

    for text in legend.get_texts():
        text.set_fontweight("normal")
        text.set_fontsize(LEGEND_FONT_SIZE)

    title = legend.get_title()
    if title is not None:
        title.set_fontweight("normal")


def style_ax(ax):
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("gray")

    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=4.0, width=1.0, pad=3)
    ax.yaxis.set_major_formatter(FuncFormatter(compact_y_tick_label))
    regular_axis_text(ax)


def set_even_eta_axis(ax, eta_values):
    eta_values = np.asarray(eta_values, dtype=float)
    x = np.arange(len(eta_values))

    ax.set_xlim(-0.5, len(eta_values) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_param_label(value) for value in eta_values],
        rotation=45,
        ha="right",
        fontsize=XTICK_LABEL_SIZE,
        fontweight="normal",
    )
    return x


def seed_file_candidates(result_dir, dataset_name, sample, seed):
    dataset_dir = os.path.join(result_dir, f"{dataset_name} {sample:.2f}")
    return [
        os.path.join(dataset_dir, f"{dataset_name} {sample:.2f} {seed}.csv"),
        os.path.join(dataset_dir, f"{dataset_name} {sample:.2f} seed_{seed} summary_results.csv"),
    ]


def normalize_seed_summary(df):
    """Normalize either the new per-seed output or the older summary_results format."""
    df = df.copy()

    if "eta" not in df.columns:
        if "mean_eta" in df.columns:
            df["eta"] = df["mean_eta"]
        else:
            df["eta"] = np.nan
            if "gamma_ratio" in df.columns:
                df.loc[df["method"] == "RSI-EC", "eta"] = df.loc[df["method"] == "RSI-EC", "gamma_ratio"]
            if "lambda" in df.columns:
                df.loc[df["method"] == "RSI-CS", "eta"] = df.loc[df["method"] == "RSI-CS", "lambda"]

    rename_map = {
        "mean_fdr": "fdp",
        "mean_power": "power",
        "mean_average_cost": "average_cost",
        "mean_n_selected": "n_selected",
        "mean_n_selected_fail": "n_selected_fail",
        "mean_n_selected_ind": "n_selected_ind",
        "mean_n_selected_pass": "n_selected_pass",
    }
    for old_name, new_name in rename_map.items():
        if new_name not in df.columns and old_name in df.columns:
            df[new_name] = df[old_name]

    required = ["method", "q", "eta", "fdp", "power", "average_cost"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    optional = ["n_selected", "n_selected_fail", "n_selected_ind", "n_selected_pass"]
    keep_cols = required + [col for col in optional if col in df.columns]
    out = df[keep_cols].copy()
    out = out.dropna(subset=["method", "q", "eta"])
    return out


def filter_eta_grid(df, eta_grid):
    if eta_grid is None or df.empty:
        return df

    eta_values = df["eta"].astype(float).to_numpy()
    keep = np.zeros(len(df), dtype=bool)
    for eta in eta_grid:
        keep |= np.isclose(eta_values, eta, rtol=1e-7, atol=1e-12)

    return df.loc[keep].copy()


def aggregate_seed_summaries(result_dir, dataset_name, sample, n_itr, seed_start=1, eta_grid=None, strict=False):
    df_list = []

    for seed in range(seed_start, seed_start + n_itr):
        file_path = None
        for candidate in seed_file_candidates(result_dir, dataset_name, sample, seed):
            if os.path.exists(candidate):
                file_path = candidate
                break

        if file_path is None:
            msg = f"[Missing] {dataset_name}, seed={seed}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        df = pd.read_csv(file_path)
        df = normalize_seed_summary(df)
        df["seed"] = seed
        df_list.append(df)

    if not df_list:
        return pd.DataFrame()

    all_df = pd.concat(df_list, ignore_index=True)
    all_df = filter_eta_grid(all_df, eta_grid)

    if all_df.empty:
        return all_df

    metric_cols = [
        col for col in [
            "fdp",
            "power",
            "average_cost",
            "n_selected",
            "n_selected_fail",
            "n_selected_ind",
            "n_selected_pass",
        ]
        if col in all_df.columns
    ]

    grouped = (
        all_df
        .groupby(["method", "q", "eta"], as_index=False, dropna=False)[metric_cols]
        .mean()
        .rename(columns={
            "fdp": "mean_fdr",
            "power": "mean_power",
            "average_cost": "mean_average_cost",
            "n_selected": "mean_n_selected",
            "n_selected_fail": "mean_n_selected_fail",
            "n_selected_ind": "mean_n_selected_ind",
            "n_selected_pass": "mean_n_selected_pass",
        })
        .sort_values(["method", "q", "eta"])
        .reset_index(drop=True)
    )
    return grouped


def get_method_q_data(df, method, q_value):
    if df is None or df.empty:
        return pd.DataFrame()

    dsub = df[(df["method"] == method) & (np.isclose(df["q"], q_value))].copy()
    if dsub.empty:
        return dsub

    return dsub.sort_values("eta").reset_index(drop=True)


def draw_empty_panel(ax, dataset_name, message="No data"):
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=PANEL_TITLE_SIZE,
        color="gray",
        fontweight="normal",
    )
    ax.set_title(dataset_name, fontsize=PANEL_TITLE_SIZE, pad=4, fontweight="normal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.7)


def plot_absolute_cell(fig, cell_spec, df, dataset_name, method, q_value, show_xlabel=False, show_ylabel=False):
    inner = cell_spec.subgridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.02)
    ax_top = fig.add_subplot(inner[0])
    ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

    dsub = get_method_q_data(df, method, q_value)
    if dsub.empty:
        draw_empty_panel(ax_top, dataset_name, "No data")
        draw_empty_panel(ax_bottom, "", "")
        return ax_top, ax_bottom, None

    eta_values = dsub["eta"].to_numpy(dtype=float)
    x = np.arange(len(dsub))

    cost = dsub["mean_average_cost"].to_numpy(dtype=float)
    power = dsub["mean_power"].to_numpy(dtype=float)
    fdp = dsub["mean_fdr"].to_numpy(dtype=float)

    ax_top.bar(x, cost, width=0.50, color=COST_COLOR, alpha=0.85, zorder=2)
    ax_top.set_title(dataset_name, fontsize=PANEL_TITLE_SIZE, pad=4, fontweight="normal")
    ax_top.set_ylim(0, nice_upper_bound(np.nanmax(cost), default=1.0))
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_top.set_ylabel("Cost" if show_ylabel else "", fontsize=AXIS_LABEL_SIZE, fontweight="normal")

    ax_fdp = ax_top.twinx()
    ax_fdp.plot(x, fdp, color=FDP_COLOR, marker="o", linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, zorder=3)
    ax_fdp.axhline(q_value, color=FDP_COLOR, linestyle="--", linewidth=TARGET_LINE_WIDTH, alpha=0.6)
    upper_fdp = max(0.10, nice_upper_bound(np.nanmax(fdp), default=0.10), 1.10 * q_value)
    ax_fdp.set_ylim(0, upper_fdp)
    ax_fdp.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_fdp.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))
    ax_fdp.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, colors=FDP_COLOR, length=4.0, width=1.0, pad=3)
    ax_fdp.set_ylabel("FDP" if not show_ylabel else "", fontsize=AXIS_LABEL_SIZE, color=FDP_COLOR, fontweight="normal")
    regular_axis_text(ax_fdp)

    ax_bottom.bar(x, power, width=0.50, color=POWER_COLOR, alpha=0.85, zorder=2)
    ax_bottom.set_ylim(0, max(0.05, nice_upper_bound(np.nanmax(power), default=1.0)))
    ax_bottom.invert_yaxis()
    ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: "" if np.isclose(y, 0) else f"{y:g}"))
    ax_bottom.set_ylabel("Power" if show_ylabel else "", fontsize=AXIS_LABEL_SIZE, fontweight="normal")

    ax_bottom.set_xlim(-0.5, len(dsub) - 0.5)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(
        [format_param_label(value) for value in eta_values],
        rotation=45,
        ha="right",
        fontsize=XTICK_LABEL_SIZE,
        fontweight="normal",
    )
    ax_bottom.set_xlabel(r"$\eta$" if show_xlabel else "", fontsize=AXIS_LABEL_SIZE, fontweight="normal")

    style_ax(ax_top)
    style_ax(ax_bottom)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", top=False)

    return ax_top, ax_bottom, ax_fdp


def make_method_absolute_figure(dataset_list, sample, q_value, method, result_dir, out_dir, n_itr, seed_start, eta_grid, strict=False):
    nrows, ncols = 3, 5
    fig = plt.figure(figsize=(36, 20))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.42, hspace=0.46)

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols

        if idx >= len(dataset_list):
            ax = fig.add_subplot(outer[row, col])
            ax.axis("off")
            continue

        dataset_name = dataset_list[idx]
        df = aggregate_seed_summaries(result_dir, dataset_name, sample, n_itr, seed_start, eta_grid, strict)
        plot_absolute_cell(
            fig=fig,
            cell_spec=outer[row, col],
            df=df,
            dataset_name=dataset_name,
            method=method,
            q_value=q_value,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )

    method_tag = method_to_tag(method)
    q_tag = q_to_tag(q_value)

    fig.subplots_adjust(left=0.045, right=0.970, top=0.955, bottom=0.215)

    legend_handles = [
        Patch(facecolor=COST_COLOR, edgecolor="none", alpha=0.85, label="Average Cost"),
        Patch(facecolor=POWER_COLOR, edgecolor="none", alpha=0.85, label="Power"),
        Line2D([0], [0], color=FDP_COLOR, marker="o", linewidth=LEGEND_LINE_WIDTH, markersize=LEGEND_MARKER_SIZE, label="FDP"),
        Line2D([0], [0], color=FDP_COLOR, linestyle="--", linewidth=TARGET_LINE_WIDTH, alpha=0.6, label=r"Target $q$"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=2.0,
        handletextpad=0.90,
        borderpad=0.90,
        handlelength=3.2,
        prop={"weight": "normal", "size": LEGEND_FONT_SIZE},
    )
    regular_legend_text(legend)

    filename = f"combined_{method_tag}_split_average_cost_{q_tag}.pdf"
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {save_path}")


def plot_delta_cell(ax, df, dataset_name, method, q_value, show_xlabel=False):
    dsub = get_method_q_data(df, method, q_value)
    if dsub.empty:
        draw_empty_panel(ax, dataset_name, "No data")
        return

    base = dsub[np.isclose(dsub["eta"].astype(float), 0.0)]
    if base.empty:
        draw_empty_panel(ax, dataset_name, r"No $\eta=0$")
        return

    eta_values = dsub["eta"].to_numpy(dtype=float)
    x = np.arange(len(eta_values))

    cost = dsub["mean_average_cost"].to_numpy(dtype=float)
    power = dsub["mean_power"].to_numpy(dtype=float)
    fdp = dsub["mean_fdr"].to_numpy(dtype=float)

    cost0 = float(base["mean_average_cost"].iloc[0])
    power0 = float(base["mean_power"].iloc[0])
    fdp0 = float(base["mean_fdr"].iloc[0])

    cost_reduction_pct = 100.0 * (cost0 - cost) / cost0 if cost0 > 0 else np.zeros_like(cost)
    delta_power_pp = 100.0 * (power - power0)
    delta_fdp_pp = 100.0 * (fdp - fdp0)

    ax.axhline(0.0, color=DELTA_ZERO_COLOR, linewidth=TARGET_LINE_WIDTH, linestyle="--", alpha=0.8)
    ax.plot(x, cost_reduction_pct, color=COST_COLOR, marker="o", linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, label="Cost reduction (%)")
    ax.plot(x, delta_power_pp, color=POWER_COLOR, marker="s", linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, label=r"$\Delta$Power (pp)")
    ax.plot(x, delta_fdp_pp, color=FDP_COLOR, marker="^", linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, label=r"$\Delta$FDP (pp)")

    ax.set_title(dataset_name, fontsize=PANEL_TITLE_SIZE, pad=4, fontweight="normal")
    ax.set_xlabel(r"$\eta$" if show_xlabel else "", fontsize=AXIS_LABEL_SIZE, fontweight="normal")
    ax.set_ylabel("", fontsize=AXIS_LABEL_SIZE, fontweight="normal")
    set_even_eta_axis(ax, eta_values)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    style_ax(ax)


def make_method_delta_figure(dataset_list, sample, q_value, method, result_dir, out_dir, n_itr, seed_start, eta_grid, strict=False):
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 20), squeeze=False)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(dataset_list):
            ax.axis("off")
            continue

        row = idx // ncols
        dataset_name = dataset_list[idx]
        df = aggregate_seed_summaries(result_dir, dataset_name, sample, n_itr, seed_start, eta_grid, strict)
        plot_delta_cell(
            ax=ax,
            df=df,
            dataset_name=dataset_name,
            method=method,
            q_value=q_value,
            show_xlabel=(row == nrows - 1),
        )

    method_tag = method_to_tag(method)
    q_tag = q_to_tag(q_value)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.045, right=0.985, top=0.955, bottom=0.215, hspace=0.56, wspace=0.52)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=2.0,
        handletextpad=0.90,
        borderpad=0.90,
        handlelength=3.2,
        prop={"weight": "normal", "size": LEGEND_FONT_SIZE},
    )
    regular_legend_text(legend)

    filename = f"combined_{method_tag}_delta_cost_power_{q_tag}.pdf"
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cost-aware RSI results by averaging seed-level files directly.")
    parser.add_argument("n_itr", type=int, help="Number of seed files to average, as in plot-settingI.py.")
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default="result-cost")
    parser.add_argument("--output_dir", type=str, default="figure-cost")
    parser.add_argument("--datasets", type=str, nargs="+", default=ALL_DATASETS)
    parser.add_argument("--q_values", type=float, nargs="+", default=[0.1,0.2,0.3,0.4,0.5])
    parser.add_argument("--methods", type=str, nargs="+", default=["RSI-EC", "RSI-CS"], choices=["RSI-EC", "RSI-CS"])
    parser.add_argument("--eta_grid", type=str, default="all", help="Comma-separated eta values to keep. Default: all values found.")
    parser.add_argument("--plot_type", type=str, default="both", choices=["absolute", "delta", "both"])
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    eta_grid = parse_eta_grid(args.eta_grid)

    out_dir = os.path.join(args.output_dir, f"sample_{args.sample:.2f}")
    os.makedirs(out_dir, exist_ok=True)

    print("[Info] script_version:", SCRIPT_VERSION)
    print("[Info] result_dir:", args.result_dir)
    print("[Info] output_dir:", out_dir)
    print("[Info] n_itr:", args.n_itr)
    print("[Info] seed_start:", args.seed_start)
    print("[Info] sample:", args.sample)
    print("[Info] datasets:", args.datasets)
    print("[Info] q_values:", args.q_values)
    print("[Info] methods:", args.methods)
    print("[Info] eta_grid:", "all" if eta_grid is None else eta_grid.tolist())
    print("[Info] plot_type:", args.plot_type)

    for q_value in args.q_values:
        for method in args.methods:
            if args.plot_type in {"absolute", "both"}:
                make_method_absolute_figure(
                    dataset_list=args.datasets,
                    sample=args.sample,
                    q_value=q_value,
                    method=method,
                    result_dir=args.result_dir,
                    out_dir=out_dir,
                    n_itr=args.n_itr,
                    seed_start=args.seed_start,
                    eta_grid=eta_grid,
                    strict=args.strict,
                )

            if args.plot_type in {"delta", "both"}:
                make_method_delta_figure(
                    dataset_list=args.datasets,
                    sample=args.sample,
                    q_value=q_value,
                    method=method,
                    result_dir=args.result_dir,
                    out_dir=out_dir,
                    n_itr=args.n_itr,
                    seed_start=args.seed_start,
                    eta_grid=eta_grid,
                    strict=args.strict,
                )

    print("All figures saved to:", out_dir)


if __name__ == "__main__":
    main()

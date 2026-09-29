import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

SCRIPT_VERSION = "three-panel-direct-seed-mean-v1"

REPRESENTATIVE_DATASETS = ['HIVPROT','OX2']

LIGHT_GRAY = "#D9D9D9"
COST_COLOR = "#5CBF60"
POWER_COLOR = "#8C6BB1"
FDP_COLOR = "#E85B5B"
FAIL_COLOR = "#D95F02"
IND_COLOR = "#7570B3"
PASS_COLOR = "#1B9E77"
ZERO_COLOR = "#666666"

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 13

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
    "font.size": TICK_LABEL_SIZE,
    "grid.color": LIGHT_GRAY,
    "grid.alpha": 0.8,
    "grid.linestyle": "-",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
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


def nice_limit_from_values(values, default=1.0, pad=0.10):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return -default, default

    lo = float(np.min(arr))
    hi = float(np.max(arr))

    if np.isclose(lo, hi):
        bump = default if np.isclose(lo, 0.0) else abs(lo) * 0.25
        return lo - bump, hi + bump

    span = hi - lo
    return lo - pad * span, hi + pad * span


def nice_upper_bound(value, default=1.0):
    if value is None or not np.isfinite(value) or value <= 0:
        return default
    return 1.10 * value


def normalize_axis_text(ax):
    ax.title.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("normal")


def normalize_legend_text(legend):
    if legend is None:
        return

    for text in legend.get_texts():
        text.set_fontweight("normal")

    title = legend.get_title()
    if title is not None:
        title.set_fontweight("normal")


def style_ax(ax):
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("gray")

    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=2.2)
    normalize_axis_text(ax)


def seed_file_candidates(result_dir, dataset_name, sample, seed):
    dataset_dir = os.path.join(result_dir, f"{dataset_name} {sample:.2f}")
    return [
        os.path.join(dataset_dir, f"{dataset_name} {sample:.2f} {seed}.csv"),
        os.path.join(dataset_dir, f"{dataset_name} {sample:.2f} seed_{seed} summary_results.csv"),
    ]


def normalize_seed_summary(df):
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

    required = [
        "method",
        "q",
        "eta",
        "fdp",
        "power",
        "average_cost",
        "n_selected_fail",
        "n_selected_ind",
        "n_selected_pass",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[required].copy()
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

    grouped = (
        all_df
        .groupby(["method", "q", "eta"], as_index=False, dropna=False)
        .mean(numeric_only=True)
        .rename(columns={
            "fdp": "mean_fdr",
            "power": "mean_power",
            "average_cost": "mean_average_cost",
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


def draw_empty_panel(ax, title, message="No data"):
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=TITLE_SIZE, color="gray")
    ax.set_title(title, fontsize=TITLE_SIZE, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.7)


def get_base_row(dsub):
    base = dsub[np.isclose(dsub["eta"].astype(float), 0.0)]
    if base.empty:
        return None
    return base.iloc[0]


def set_eta_ticks(ax, x, eta_values):
    ax.set_xticks(x)
    ax.set_xticklabels([format_param_label(value) for value in eta_values], rotation=45, ha="right")


def first_panel_legend_handles():
    return [
        Line2D([0], [0], color=COST_COLOR, linewidth=7, alpha=0.85, label="Average cost"),
        Line2D([0], [0], color=POWER_COLOR, linewidth=7, alpha=0.85, label="Power"),
        Line2D([0], [0], color=FDP_COLOR, marker="o", linewidth=1.0, markersize=4, label="FDP"),
    ]


def plot_power_cost_hist_cell(fig, cell_spec, dsub, dataset_name, show_legend=False):
    inner = cell_spec.subgridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.02)
    ax_top = fig.add_subplot(inner[0])
    ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

    if dsub.empty:
        draw_empty_panel(ax_top, f"{dataset_name}: cost, FDP, power", "No data")
        draw_empty_panel(ax_bottom, "", "")
        return ax_top, ax_bottom, None

    x = np.arange(len(dsub))
    eta_values = dsub["eta"].to_numpy(dtype=float)

    cost = dsub["mean_average_cost"].to_numpy(dtype=float)
    power = dsub["mean_power"].to_numpy(dtype=float)
    fdp = dsub["mean_fdr"].to_numpy(dtype=float)

    ax_top.bar(x, cost, width=0.50, color=COST_COLOR, alpha=0.85, zorder=2)
    ax_top.set_title(f"{dataset_name}: cost, FDP, power", fontsize=TITLE_SIZE, pad=4)
    ax_top.set_ylim(0, nice_upper_bound(np.nanmax(cost), default=1.0))
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_top.set_ylabel("Cost")

    ax_fdp = ax_top.twinx()
    ax_fdp.plot(x, fdp, color=FDP_COLOR, marker="o", linewidth=1.0, markersize=2.8, zorder=3)
    ax_fdp.set_ylim(0, max(0.10, nice_upper_bound(np.nanmax(fdp), default=0.10)))
    ax_fdp.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_fdp.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))
    ax_fdp.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, colors=FDP_COLOR, length=2.0)
    ax_fdp.set_ylabel("FDP", color=FDP_COLOR)
    normalize_axis_text(ax_fdp)

    ax_bottom.bar(x, power, width=0.50, color=POWER_COLOR, alpha=0.85, zorder=2)
    ax_bottom.set_ylim(0, max(0.05, nice_upper_bound(np.nanmax(power), default=1.0)))
    ax_bottom.invert_yaxis()
    ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: "" if np.isclose(y, 0) else f"{y:g}"))
    ax_bottom.set_ylabel("Power")
    ax_bottom.set_xlabel(r"$\eta$")

    set_eta_ticks(ax_bottom, x, eta_values)

    style_ax(ax_top)
    style_ax(ax_bottom)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", top=False)

    if show_legend:
        legend = ax_bottom.legend(
            handles=first_panel_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, -1.18),
            ncol=3,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            columnspacing=1.00,
            handletextpad=0.45,
            borderpad=0.45,
        )
        normalize_legend_text(legend)

    return ax_top, ax_bottom, ax_fdp


def plot_delta_metrics(ax, dsub, dataset_name, show_legend=False):
    base = get_base_row(dsub)
    if base is None:
        draw_empty_panel(ax, f"{dataset_name}: delta metrics", r"No $\eta=0$")
        return

    eta_values = dsub["eta"].to_numpy(dtype=float)
    x = np.arange(len(eta_values))

    cost = dsub["mean_average_cost"].to_numpy(dtype=float)
    power = dsub["mean_power"].to_numpy(dtype=float)
    fdp = dsub["mean_fdr"].to_numpy(dtype=float)

    cost0 = float(base["mean_average_cost"])
    power0 = float(base["mean_power"])
    fdp0 = float(base["mean_fdr"])

    delta_cost_reduction_pct = 100.0 * (cost0 - cost) / cost0 if cost0 > 0 else np.zeros_like(cost)
    delta_power_pp = 100.0 * (power - power0)
    delta_fdp_pp = 100.0 * (fdp - fdp0)

    ax.axhline(0.0, color=ZERO_COLOR, linewidth=0.9, linestyle="--", alpha=0.8)
    ax.plot(x, delta_cost_reduction_pct, color=COST_COLOR, marker="o", linewidth=1.35, markersize=4.0, label="Cost reduction (%)")
    ax.plot(x, delta_power_pp, color=POWER_COLOR, marker="s", linewidth=1.35, markersize=4.0, label=r"$\Delta$Power (pp)")
    ax.plot(x, delta_fdp_pp, color=FDP_COLOR, marker="^", linewidth=1.35, markersize=4.0, label=r"$\Delta$FDP (pp)")

    ymin, ymax = nice_limit_from_values(np.r_[delta_cost_reduction_pct, delta_power_pp, delta_fdp_pp], default=5.0)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(f"{dataset_name}: delta metrics", fontsize=TITLE_SIZE, pad=4)
    ax.set_ylabel(r"Difference from $\eta=0$")
    ax.set_xlabel(r"$\eta$")
    set_eta_ticks(ax, x, eta_values)
    style_ax(ax)

    if show_legend:
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=3,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            columnspacing=0.85,
            handletextpad=0.45,
            borderpad=0.45,
        )
        normalize_legend_text(legend)


def plot_delta_selected_counts(ax, dsub, dataset_name, show_legend=False):
    base = get_base_row(dsub)
    if base is None:
        draw_empty_panel(ax, f"{dataset_name}: selected-count changes", r"No $\eta=0$")
        return

    eta_values = dsub["eta"].to_numpy(dtype=float)
    x = np.arange(len(eta_values))

    delta_fail = dsub["mean_n_selected_fail"].to_numpy(dtype=float) - float(base["mean_n_selected_fail"])
    delta_ind = dsub["mean_n_selected_ind"].to_numpy(dtype=float) - float(base["mean_n_selected_ind"])
    delta_pass = dsub["mean_n_selected_pass"].to_numpy(dtype=float) - float(base["mean_n_selected_pass"])

    ax.axhline(0.0, color=ZERO_COLOR, linewidth=0.9, linestyle="--", alpha=0.8)
    ax.plot(x, delta_fail, color=FAIL_COLOR, marker="o", linewidth=1.35, markersize=4.0, label=r"$\Delta n_{\mathrm{fail}}$")
    ax.plot(x, delta_ind, color=IND_COLOR, marker="s", linewidth=1.35, markersize=4.0, label=r"$\Delta n_{\mathrm{ind}}$")
    ax.plot(x, delta_pass, color=PASS_COLOR, marker="^", linewidth=1.35, markersize=4.0, label=r"$\Delta n_{\mathrm{pass}}$")

    ymin, ymax = nice_limit_from_values(np.r_[delta_fail, delta_ind, delta_pass], default=10.0)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(f"{dataset_name}: selected-count changes", fontsize=TITLE_SIZE, pad=4)
    ax.set_ylabel("Change in mean selected count")
    ax.set_xlabel(r"$\eta$")
    set_eta_ticks(ax, x, eta_values)
    style_ax(ax)

    if show_legend:
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=3,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            columnspacing=0.85,
            handletextpad=0.45,
            borderpad=0.45,
        )
        normalize_legend_text(legend)


def make_three_panel_rows_figure(dataset_list, sample, q_value, method, result_dir, out_dir, n_itr, seed_start, eta_grid, strict=False, file_format="pdf", dpi=300):
    nrows = len(dataset_list)
    if nrows <= 0:
        raise ValueError("At least one dataset must be provided.")

    row_height = 6
    fig_height = max(6.2, row_height * nrows)
    fig_width = 22.0

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(nrows, 3, wspace=0.36, hspace=0.48)

    for row, dataset_name in enumerate(dataset_list):
        df = aggregate_seed_summaries(result_dir, dataset_name, sample, n_itr, seed_start, eta_grid, strict)
        dsub = get_method_q_data(df, method, q_value)

        plot_power_cost_hist_cell(
            fig,
            outer[row, 0],
            dsub,
            dataset_name,
            show_legend=(row == nrows - 1),
        )

        ax_delta = fig.add_subplot(outer[row, 1])
        ax_counts = fig.add_subplot(outer[row, 2])

        if dsub.empty:
            draw_empty_panel(ax_delta, f"{dataset_name}: delta metrics", "No data")
            draw_empty_panel(ax_counts, f"{dataset_name}: selected-count changes", "No data")
            continue

        plot_delta_metrics(ax_delta, dsub, dataset_name, show_legend=(row == nrows - 1))
        plot_delta_selected_counts(ax_counts, dsub, dataset_name, show_legend=(row == nrows - 1))

    method_tag = method_to_tag(method)
    q_tag = q_to_tag(q_value)
    dataset_tag = "_".join(dataset_list).replace("/", "-").replace(" ", "_")

    top_margin_in = 1.05
    bottom_margin_in = 2.25
    fig.subplots_adjust(
        left=0.055,
        right=0.965,
        top=1.0 - top_margin_in / fig_height,
        bottom=bottom_margin_in / fig_height,
        hspace=0.68,
        wspace=0.46,
    )

    filename = f"{method_tag}_{dataset_tag}_three_panel_rows_{q_tag}.{file_format}"
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[Saved] {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot representative three-panel cost-aware RSI figures.")
    parser.add_argument("--n_itr", type=int, help="Number of seed files to average, as in plot-settingI.py.")
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default="result-cost")
    parser.add_argument("--output_dir", type=str, default="figure-cost-three-panel")
    parser.add_argument("--datasets", type=str, nargs="+", default=REPRESENTATIVE_DATASETS)
    parser.add_argument("--q_values", type=float, nargs="+", default=[0.2])
    parser.add_argument("--methods", type=str, nargs="+", default=["RSI-EC", "RSI-CS"], choices=["RSI-EC", "RSI-CS"])
    parser.add_argument("--eta_grid", type=str, default="all", help="Comma-separated eta values to keep. Default: all values found.")
    parser.add_argument("--file_format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
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
    print("[Info] file_format:", args.file_format)

    for q_value in args.q_values:
        for method in args.methods:
            make_three_panel_rows_figure(
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
                file_format=args.file_format,
                dpi=args.dpi,
            )

    print("All figures saved to:", out_dir)


if __name__ == "__main__":
    main()

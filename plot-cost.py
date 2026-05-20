import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_VERSION = 'method-separated-v13-even-eta-axis'
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ============================================================
# Method-separated 5 x 3 figures for paper-final-cost-0505.py
#
# For each q and each method, this script makes one 15-panel figure:
#   - RSI-EC: 15 datasets in a 5 x 3 layout
#   - RSI-CS: 15 datasets in a 5 x 3 layout
#
# Expected input file for each dataset:
#   <result_dir>/<dataset> <sample>/<dataset> <sample> aggregated_over_seeds.csv
#
# Example:
#   result-cost-0505/3A4 1.00/3A4 1.00 aggregated_over_seeds.csv
#
# Figure types:
#   absolute: average cost bars + FDR line, power bars below
#   delta: changes relative to eta=0
#   both: generate both
# ============================================================

# ---------- datasets ----------
ALL_DATASETS = [
    '3A4', 'CB1', 'DPP4',
    'HIVINT', 'HIVPROT', 'LOGD',
    'METAB', 'NK1', 'OX1',
    'OX2', 'PGP', 'PPB',
    'RAT_F', 'TDI', 'THROMBIN'
]

# ---------- colors ----------
LIGHT_GRAY = '#D9D9D9'
COST_COLOR = '#5CBF60'
POWER_COLOR = '#8C6BB1'
FDR_COLOR = '#E85B5B'
DELTA_ZERO_COLOR = '#666666'

# ---------- font sizes ----------
TITLE_SIZE = 26
AXIS_LABEL_SIZE = 25
TICK_LABEL_SIZE = 23
XTICK_LABEL_SIZE = 23
PANEL_TITLE_SIZE = 36
SUPTITLE_SIZE = 36
LEGEND_FONT_SIZE = 36

# ---------- line and marker sizes ----------
MAIN_LINE_WIDTH = 3.2
TARGET_LINE_WIDTH = 2.5
MARKER_SIZE = 7.5
LEGEND_LINE_WIDTH = 5.2
LEGEND_MARKER_SIZE = 13.0

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': '#666666',
    'axes.labelsize': AXIS_LABEL_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
    'grid.color': LIGHT_GRAY,
    'grid.alpha': 0.8,
    'grid.linestyle': '-',
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'legend.fontsize': LEGEND_FONT_SIZE,
})


# ============================================================
# Helpers
# ============================================================
def parse_eta_grid(grid_string):
    values = []
    for item in grid_string.split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    if len(values) == 0:
        raise ValueError('eta_grid must contain at least one numeric value.')
    return np.array(values, dtype=float)


def q_to_tag(q):
    return f'q{int(round(q * 10)):02d}'


def method_to_tag(method):
    return method.replace('RSI-', '').lower()


def format_param_label(v):
    if pd.isna(v):
        return ''
    v = float(v)
    if v.is_integer():
        return str(int(v))
    return f'{v:g}'


def set_even_eta_axis(ax, eta):
    """Use evenly spaced x positions while displaying the true eta values."""
    eta = np.asarray(eta, dtype=float)
    x = np.arange(len(eta))
    ax.set_xlim(-0.5, len(eta) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_param_label(v) for v in eta],
        rotation=45,
        ha='right',
        fontsize=XTICK_LABEL_SIZE,
        fontweight='normal'
    )
    return x


def compact_y_tick_label(y, pos):
    """Compact y-axis labels to reduce horizontal overlap across panels."""
    if not np.isfinite(y):
        return ''
    ay = abs(y)
    if ay >= 10:
        return f'{y:.0f}'
    if ay >= 1:
        return f'{y:.1f}'.rstrip('0').rstrip('.')
    if ay >= 0.1:
        return f'{y:.2f}'.rstrip('0').rstrip('.')
    return f'{y:.2f}'


def nice_upper_bound(x, default=1.0):
    if x is None or not np.isfinite(x) or x <= 0:
        return default
    return 1.10 * x


def regular_axis_text(ax):
    """Use regular-weight axis title, labels, and tick-label numbers."""
    ax.title.set_fontweight('normal')
    ax.xaxis.label.set_fontweight('normal')
    ax.yaxis.label.set_fontweight('normal')
    ax.title.set_fontsize(PANEL_TITLE_SIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(TICK_LABEL_SIZE)
        tick_label.set_fontweight('normal')


def regular_tick_labels(ax):
    """Backward-compatible helper for tick labels only."""
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(TICK_LABEL_SIZE)
        tick_label.set_fontweight('normal')


def regular_legend_text(legend):
    """Use regular-weight legend text."""
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight('normal')
        text.set_fontsize(LEGEND_FONT_SIZE)
    title = legend.get_title()
    if title is not None:
        title.set_fontweight('normal')


def style_ax(ax):
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor('gray')
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE, length=4.0, width=1.0, pad=3)
    ax.yaxis.set_major_formatter(FuncFormatter(compact_y_tick_label))
    regular_axis_text(ax)


def load_dataset_summary(result_dir, dataset_name, sample, strict=False):
    csv_path = os.path.join(
        result_dir,
        f'{dataset_name} {sample:.2f}',
        f'{dataset_name} {sample:.2f} aggregated_over_seeds.csv'
    )

    if not os.path.exists(csv_path):
        msg = f'[Missing] {csv_path}'
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        msg = f'[Empty] {csv_path}'
        if strict:
            raise ValueError(msg)
        print(msg)
        return None

    required_cols = {'method', 'q', 'mean_fdr', 'mean_power', 'mean_average_cost'}
    missing = required_cols.difference(df.columns)
    if missing:
        msg = f'{csv_path} is missing required columns: {sorted(missing)}'
        if strict:
            raise ValueError(msg)
        print('[Invalid]', msg)
        return None

    return df


def choose_eta_column(dsub, method):
    """
    paper-final-cost-0505.py stores eta in historical columns:
      - mean_eta: preferred after aggregation
      - gamma_ratio or lambda_ec: eta for RSI-EC
      - lambda: eta for RSI-CS
      - eta: possible in non-aggregated files
    The plot always labels the x-axis as eta.
    """
    if 'mean_eta' in dsub.columns and dsub['mean_eta'].notna().any():
        return 'mean_eta', r'$\eta$'

    if 'eta' in dsub.columns and dsub['eta'].notna().any():
        return 'eta', r'$\eta$'

    if method == 'RSI-EC':
        if 'gamma_ratio' in dsub.columns and dsub['gamma_ratio'].notna().any():
            return 'gamma_ratio', r'$\eta$'
        if 'lambda_ec' in dsub.columns and dsub['lambda_ec'].notna().any():
            return 'lambda_ec', r'$\eta$'
        raise ValueError('No eta column found for RSI-EC. Expected mean_eta, eta, gamma_ratio, or lambda_ec.')

    if method == 'RSI-CS':
        if 'lambda' in dsub.columns and dsub['lambda'].notna().any():
            return 'lambda', r'$\eta$'
        raise ValueError('No eta column found for RSI-CS. Expected mean_eta, eta, or lambda.')

    raise ValueError("method must be 'RSI-EC' or 'RSI-CS'.")


def grid_order_value(v, grid):
    v = float(v)
    for j, g in enumerate(grid):
        if np.isclose(v, g, rtol=1e-7, atol=1e-12):
            return j
    return np.nan


def filter_to_eta_grid(dsub, eta_col, eta_grid):
    if dsub is None or dsub.empty:
        return dsub

    values = dsub[eta_col].astype(float).to_numpy()
    keep = np.zeros(len(dsub), dtype=bool)
    for g in eta_grid:
        keep |= np.isclose(values, g, rtol=1e-7, atol=1e-12)

    out = dsub.loc[keep].copy()
    if out.empty:
        return out

    out['_eta_order'] = out[eta_col].apply(lambda v: grid_order_value(v, eta_grid))
    out = out.sort_values('_eta_order')
    return out.drop(columns=['_eta_order'])


def get_method_q_data(df, method, q_value, eta_grid):
    if df is None:
        return pd.DataFrame(), None, None

    dsub = df[(df['method'] == method) & (np.isclose(df['q'], q_value))].copy()
    if dsub.empty:
        return dsub, None, None

    eta_col, eta_label = choose_eta_column(dsub, method)
    dsub = dsub[dsub[eta_col].notna()].copy()
    dsub = filter_to_eta_grid(dsub, eta_col, eta_grid)

    if dsub.empty:
        return dsub, eta_col, eta_label

    # Safety: if there are duplicated rows after aggregation, average them.
    metric_cols = [
        col for col in [
            'mean_fdr', 'mean_power', 'mean_average_cost',
            'mean_n_selected', 'mean_n_selected_fail',
            'mean_n_selected_ind', 'mean_n_selected_pass'
        ]
        if col in dsub.columns
    ]
    if dsub.duplicated(subset=[eta_col]).any():
        dsub = dsub.groupby(eta_col, as_index=False)[metric_cols].mean()
        dsub['_eta_order'] = dsub[eta_col].apply(lambda v: grid_order_value(v, eta_grid))
        dsub = dsub.sort_values('_eta_order').drop(columns=['_eta_order'])

    return dsub, eta_col, eta_label


def draw_empty_panel(ax, dataset_name, message='No data'):
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=PANEL_TITLE_SIZE, color='gray', fontweight='normal')
    ax.set_title(dataset_name, fontsize=PANEL_TITLE_SIZE, pad=4, fontweight='normal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.7)


# ============================================================
# Absolute figure: one method per figure, 15 panels = 5 x 3
# ============================================================
def plot_absolute_cell(fig, cell_spec, df, dataset_name, method, q_value, eta_grid, show_xlabel=False, show_ylabel=False):
    inner = cell_spec.subgridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.02)
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

    dsub, eta_col, eta_label = get_method_q_data(df, method, q_value, eta_grid)

    if dsub.empty:
        draw_empty_panel(ax_top, dataset_name, 'No data')
        draw_empty_panel(ax_bot, '', '')
        return ax_top, ax_bot, None

    x = np.arange(len(dsub))
    xticklabels = [format_param_label(v) for v in dsub[eta_col].to_numpy()]

    cost = dsub['mean_average_cost'].to_numpy(dtype=float)
    power = dsub['mean_power'].to_numpy(dtype=float)
    fdr = dsub['mean_fdr'].to_numpy(dtype=float)

    # Top: average cost bars.
    ax_top.bar(x, cost, width=0.50, color=COST_COLOR, alpha=0.85, zorder=2)
    ax_top.set_title(dataset_name, fontsize=PANEL_TITLE_SIZE, pad=4, fontweight='normal')
    ax_top.set_ylim(0, nice_upper_bound(np.nanmax(cost), default=1.0))
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_top.set_ylabel('Cost' if show_ylabel else '', fontsize=AXIS_LABEL_SIZE, fontweight='normal')

    # FDR on right axis.
    ax_fdr = ax_top.twinx()
    ax_fdr.plot(x, fdr, color=FDR_COLOR, marker='o', linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, zorder=3)
    ax_fdr.axhline(q_value, color=FDR_COLOR, linestyle='--', linewidth=TARGET_LINE_WIDTH, alpha=0.6)
    upper_fdr = max(0.10, nice_upper_bound(np.nanmax(fdr), default=0.10), 1.10 * q_value)
    ax_fdr.set_ylim(0, upper_fdr)
    ax_fdr.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_fdr.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.2f}'))
    ax_fdr.tick_params(axis='y', labelsize=TICK_LABEL_SIZE, colors=FDR_COLOR, length=4.0, width=1.0, pad=3)
    ax_fdr.set_ylabel('FDP' if not show_ylabel else '', fontsize=AXIS_LABEL_SIZE, color=FDR_COLOR, fontweight='normal')
    regular_axis_text(ax_fdr)
    regular_axis_text(ax_fdr)

    # Bottom: power bars, inverted as in your original style.
    ax_bot.bar(x, power, width=0.50, color=POWER_COLOR, alpha=0.85, zorder=2)
    pmax = max(0.05, nice_upper_bound(np.nanmax(power), default=1.0))
    ax_bot.set_ylim(0, pmax)
    ax_bot.invert_yaxis()
    ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_bot.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: '' if np.isclose(y, 0) else f'{y:g}'))
    ax_bot.set_ylabel('Power' if show_ylabel else '', fontsize=AXIS_LABEL_SIZE, fontweight='normal')

    ax_bot.set_xlim(-0.5, len(dsub) - 0.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=XTICK_LABEL_SIZE, fontweight='normal')
    ax_bot.set_xlabel(eta_label if show_xlabel else '', fontsize=AXIS_LABEL_SIZE, fontweight='normal')

    style_ax(ax_top)
    style_ax(ax_bot)
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_bot.tick_params(axis='x', top=False)

    return ax_top, ax_bot, ax_fdr


def make_method_absolute_figure(dataset_list, sample, q_value, method, result_dir, out_dir, eta_grid, strict=False):
    nrows, ncols = 3, 5
    fig = plt.figure(figsize=(36, 20))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.42, hspace=0.46)

    for idx, dataset_name in enumerate(dataset_list):
        row = idx // ncols
        col = idx % ncols
        df = load_dataset_summary(result_dir, dataset_name, sample, strict=strict)
        plot_absolute_cell(
            fig=fig,
            cell_spec=outer[row, col],
            df=df,
            dataset_name=dataset_name,
            method=method,
            q_value=q_value,
            eta_grid=eta_grid,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )

    method_tag = method_to_tag(method)
    q_tag = q_to_tag(q_value)

    fig.subplots_adjust(left=0.045, right=0.970, top=0.955, bottom=0.215)

    legend_handles = [
        Patch(facecolor=COST_COLOR, edgecolor='none', alpha=0.85, label='Average Cost'),
        Patch(facecolor=POWER_COLOR, edgecolor='none', alpha=0.85, label='Power'),
        Line2D([0], [0], color=FDR_COLOR, marker='o', linewidth=LEGEND_LINE_WIDTH, markersize=LEGEND_MARKER_SIZE, label='FDP'),
        Line2D([0], [0], color=FDR_COLOR, linestyle='--', linewidth=TARGET_LINE_WIDTH, alpha=0.6, label=r'Target $q$'),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=2.0,
        handletextpad=0.90,
        borderpad=0.90,
        handlelength=3.2,
        prop={'weight': 'normal', 'size': LEGEND_FONT_SIZE},
    )
    regular_legend_text(legend)

    filename = f'combined_{method_tag}_split_average_cost_{q_tag}.pdf'
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[Saved] {save_path}')


# ============================================================
# Delta figure: one method per figure, 15 panels = 5 x 3
# ============================================================
def plot_delta_cell(ax, df, dataset_name, method, q_value, eta_grid, show_xlabel=False, show_ylabel=False):
    dsub, eta_col, eta_label = get_method_q_data(df, method, q_value, eta_grid)

    if dsub.empty:
        draw_empty_panel(ax, dataset_name, 'No data')
        return

    base = dsub[np.isclose(dsub[eta_col].astype(float), 0.0)]
    if base.empty:
        draw_empty_panel(ax, dataset_name, 'No eta=0')
        return

    eta = dsub[eta_col].to_numpy(dtype=float)
    # Plot the nonuniform eta grid at evenly spaced positions.
    # This avoids compressing small eta values near zero when the raw grid is
    # something like 0, 0.001, 0.002, ..., 0.5.
    x = np.arange(len(eta))

    cost = dsub['mean_average_cost'].to_numpy(dtype=float)
    power = dsub['mean_power'].to_numpy(dtype=float)
    fdr = dsub['mean_fdr'].to_numpy(dtype=float)

    cost0 = float(base['mean_average_cost'].iloc[0])
    power0 = float(base['mean_power'].iloc[0])
    fdr0 = float(base['mean_fdr'].iloc[0])

    if cost0 > 0:
        cost_reduction_pct = 100.0 * (cost0 - cost) / cost0
    else:
        cost_reduction_pct = np.zeros_like(cost)

    # Percentage-point changes for power and FDR.
    delta_power_pp = 100.0 * (power - power0)
    delta_fdr_pp = 100.0 * (fdr - fdr0)

    ax.axhline(0.0, color=DELTA_ZERO_COLOR, linewidth=TARGET_LINE_WIDTH, linestyle='--', alpha=0.8)
    ax.plot(x, cost_reduction_pct, color=COST_COLOR, marker='o', linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, label='Cost reduction (%)')
    ax.plot(x, delta_power_pp, color=POWER_COLOR, marker='s', linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, label=r'$\Delta$Power (pp)')
    ax.plot(x, delta_fdr_pp, color=FDR_COLOR, marker='^', linewidth=MAIN_LINE_WIDTH, markersize=MARKER_SIZE, label=r'$\Delta$FDP (pp)')

    ax.set_title(dataset_name, fontsize=PANEL_TITLE_SIZE, pad=4, fontweight='normal')
    ax.set_xlabel(eta_label if show_xlabel else '', fontsize=AXIS_LABEL_SIZE, fontweight='normal')
    # The y-axis meaning is explained in the paper caption, so avoid
    # repeating the long label on the left column of the 15-panel figure.
    ax.set_ylabel('', fontsize=AXIS_LABEL_SIZE, fontweight='normal')
    set_even_eta_axis(ax, eta)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    style_ax(ax)


def make_method_delta_figure(dataset_list, sample, q_value, method, result_dir, out_dir, eta_grid, strict=False):
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 20), squeeze=False)

    for idx, dataset_name in enumerate(dataset_list):
        row = idx // ncols
        col = idx % ncols
        df = load_dataset_summary(result_dir, dataset_name, sample, strict=strict)
        plot_delta_cell(
            ax=axes[row, col],
            df=df,
            dataset_name=dataset_name,
            method=method,
            q_value=q_value,
            eta_grid=eta_grid,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )

    method_tag = method_to_tag(method)
    q_tag = q_to_tag(q_value)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.045, right=0.985, top=0.955, bottom=0.215, hspace=0.56, wspace=0.52)
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=2.0,
        handletextpad=0.90,
        borderpad=0.90,
        handlelength=3.2,
        prop={'weight': 'normal', 'size': LEGEND_FONT_SIZE},
    )
    regular_legend_text(legend)

    filename = f'combined_{method_tag}_delta_cost_power_{q_tag}.pdf'
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[Saved] {save_path}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=float, help='Sample ratio, e.g. 0.10 or 1.00')
    parser.add_argument('--result_dir', type=str, default='result-cost-0505',
                        help='Directory containing per-dataset aggregated_over_seeds.csv files')
    parser.add_argument('--output_dir', type=str, default='figure-cost-0505',
                        help='Directory to save generated figures')
    parser.add_argument('--datasets', type=str, nargs='+', default=ALL_DATASETS,
                        help='Datasets to plot. Default: all 15 QSAR datasets')
    parser.add_argument('--q_values', type=float, nargs='+', default=[0.2],
                        help='Nominal FDR levels to plot. Default: 0.2')
    parser.add_argument('--methods', type=str, nargs='+', default=['RSI-EC', 'RSI-CS'],
                        choices=['RSI-EC', 'RSI-CS'],
                        help='Methods to plot. Default: RSI-EC RSI-CS')
    parser.add_argument('--eta_grid', type=str,
                        default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated eta grid. Default: 0,0.1,...,1.0')
    parser.add_argument('--plot_type', type=str, default='both',
                        choices=['absolute', 'delta', 'both'],
                        help='Which figure type to generate. Default: both')
    parser.add_argument('--strict', action='store_true',
                        help='Raise an error if any dataset file is missing or invalid. Default: draw missing panels.')
    args = parser.parse_args()

    sample = args.sample
    eta_grid = parse_eta_grid(args.eta_grid)

    out_dir = os.path.join(args.output_dir, f'sample_{sample:.2f}')
    os.makedirs(out_dir, exist_ok=True)

    print('[Info] script_version:', SCRIPT_VERSION)
    print('[Info] result_dir:', args.result_dir)
    print('[Info] output_dir:', out_dir)
    print('[Info] datasets:', args.datasets)
    print('[Info] q_values:', args.q_values)
    print('[Info] methods:', args.methods)
    print('[Info] eta_grid:', eta_grid.tolist())
    print('[Info] plot_type:', args.plot_type)

    if len(args.datasets) != 15:
        print(f'[Warning] Number of datasets is {len(args.datasets)}, not 15. The layout is still 5 x 3; unused panels are not created.')

    for q in args.q_values:
        for method in args.methods:
            if args.plot_type in ['absolute', 'both']:
                make_method_absolute_figure(
                    dataset_list=args.datasets,
                    sample=sample,
                    q_value=q,
                    method=method,
                    result_dir=args.result_dir,
                    out_dir=out_dir,
                    eta_grid=eta_grid,
                    strict=args.strict,
                )

            if args.plot_type in ['delta', 'both']:
                make_method_delta_figure(
                    dataset_list=args.datasets,
                    sample=sample,
                    q_value=q,
                    method=method,
                    result_dir=args.result_dir,
                    out_dir=out_dir,
                    eta_grid=eta_grid,
                    strict=args.strict,
                )

    print('All figures saved to:', out_dir)


if __name__ == '__main__':
    main()

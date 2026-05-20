import argparse
import os

import numpy as np

SCRIPT_VERSION = "three-panel-hist-v12-even-eta-axis"
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ============================================================
# Three-panel row figures for paper-final-cost-0505.py
#
# For each q and each method, this script makes one figure in which
# each dataset occupies one row and the three columns are:
#   (1) Power--cost trade-off as eta varies
#   (2) Delta metrics relative to eta = 0
#   (3) Delta selected-region counts relative to eta = 0
#
# Expected input file for each dataset:
#   <result_dir>/<dataset> <sample>/<dataset> <sample> aggregated_over_seeds.csv
#
# Example:
#   result-cost-0505/3A4 1.00/3A4 1.00 aggregated_over_seeds.csv
#
# Example commands:
#   python plot-cost-0505-15-three-panel-hist.py 1.00 --datasets 3A4 --methods RSI-EC --q_values 0.2
#   python plot-cost-0505-15-three-panel-hist.py 1.00 --datasets 3A4 DPP4 HIVPROT OX2 PPB --methods RSI-CS --q_values 0.2 0.3
#   python plot-cost-0505-15-three-panel-hist.py 1.00 --methods RSI-EC RSI-CS --q_values 0.2
# ============================================================

# ALL_DATASETS = [
#     '3A4', 'CB1', 'DPP4',
#     'HIVINT', 'HIVPROT', 'LOGD',
#     'METAB', 'NK1', 'OX1',
#     'OX2', 'PGP', 'PPB',
#     'RAT_F', 'TDI', 'THROMBIN'
# ]
# ALL_DATASETS = [
#     'PPB','OX2','DPP4', 'CB1'
# ]
ALL_DATASETS = [
    'LOGD', 'RAT_F', 'HIVPROT','OX2'
]


# ---------- colors ----------
LIGHT_GRAY = '#D9D9D9'
COST_COLOR = '#5CBF60'
POWER_COLOR = '#8C6BB1'
FDR_COLOR = '#E85B5B'
FAIL_COLOR = '#D95F02'
IND_COLOR = '#7570B3'
PASS_COLOR = '#1B9E77'
ZERO_COLOR = '#666666'

# ---------- font sizes ----------
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 13
SUPTITLE_SIZE = 20

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
    'font.size': TICK_LABEL_SIZE,
    'grid.color': LIGHT_GRAY,
    'grid.alpha': 0.8,
    'grid.linestyle': '-',
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
})


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


def nice_limit_from_values(values, default=1.0, pad=0.10):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-default, default)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        bump = default if np.isclose(lo, 0.0) else abs(lo) * 0.25
        return (lo - bump, hi + bump)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def nice_upper_bound(x, default=1.0):
    if x is None or not np.isfinite(x) or x <= 0:
        return default
    return 1.10 * x


def normalize_axis_text(ax):
    """Keep all visible text attached to an axis at normal weight."""
    ax.title.set_fontweight('normal')
    ax.xaxis.label.set_fontweight('normal')
    ax.yaxis.label.set_fontweight('normal')
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('normal')


def normalize_legend_text(legend):
    """Keep legend text at normal weight."""
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight('normal')
    title = legend.get_title()
    if title is not None:
        title.set_fontweight('normal')


def style_ax(ax):
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor('gray')
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE, length=2.2)
    normalize_axis_text(ax)


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

    required_cols = {
        'method', 'q', 'mean_fdr', 'mean_power', 'mean_average_cost',
        'mean_n_selected_fail', 'mean_n_selected_ind', 'mean_n_selected_pass'
    }
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
    The score parameter is plotted as eta regardless of how it was stored.
      - eta: preferred after aggregation if eta is a grouping column
      - mean_eta: older aggregated files
      - gamma_ratio or lambda_ec: historical EC columns
      - lambda: historical CS column
    """
    if 'eta' in dsub.columns and dsub['eta'].notna().any():
        return 'eta', r'$\eta$'

    if 'mean_eta' in dsub.columns and dsub['mean_eta'].notna().any():
        return 'mean_eta', r'$\eta$'

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

    metric_cols = [
        col for col in [
            'mean_fdr', 'mean_power', 'mean_average_cost',
            'mean_n_selected', 'mean_n_selected_fail',
            'mean_n_selected_ind', 'mean_n_selected_pass'
        ]
        if col in dsub.columns
    ]

    # Safety: average duplicated eta rows if they occur.
    if dsub.duplicated(subset=[eta_col]).any():
        dsub = dsub.groupby(eta_col, as_index=False)[metric_cols].mean()
        dsub['_eta_order'] = dsub[eta_col].apply(lambda v: grid_order_value(v, eta_grid))
        dsub = dsub.sort_values('_eta_order').drop(columns=['_eta_order'])

    return dsub, eta_col, eta_label


def draw_empty_panel(ax, title, message='No data'):
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=TITLE_SIZE, color='gray')
    ax.set_title(title, fontsize=TITLE_SIZE, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.7)


def get_base_row(dsub, eta_col):
    base = dsub[np.isclose(dsub[eta_col].astype(float), 0.0)]
    if base.empty:
        return None
    return base.iloc[0]


def set_eta_ticks(ax, x, eta):
    """Use evenly spaced x positions, while displaying the true eta values."""
    ax.set_xticks(x)
    ax.set_xticklabels([format_param_label(v) for v in eta], rotation=45, ha='right')


# ============================================================
# Panel 1: same split histogram/bar plot as plot-cost-0505-15.py
# ============================================================
def plot_power_cost_hist_cell(fig, cell_spec, dsub, eta_col, eta_label, dataset_name, q_value, show_xlabel=True, show_legend=False):
    """
    Reuse the original split bar/histogram style:
      - top panel: average cost bars + FDR line on the right axis
      - bottom panel: power bars, inverted vertically
    """
    inner = cell_spec.subgridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.02)
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

    if dsub.empty:
        draw_empty_panel(ax_top, f'{dataset_name}: cost, FDP, power', 'No data')
        draw_empty_panel(ax_bot, '', '')
        return ax_top, ax_bot, None

    x = np.arange(len(dsub))
    xticklabels = [format_param_label(v) for v in dsub[eta_col].to_numpy()]

    cost = dsub['mean_average_cost'].to_numpy(dtype=float)
    power = dsub['mean_power'].to_numpy(dtype=float)
    fdr = dsub['mean_fdr'].to_numpy(dtype=float)

    # Top: average cost bars.
    ax_top.bar(x, cost, width=0.50, color=COST_COLOR, alpha=0.85, zorder=2)
    ax_top.set_title(f'{dataset_name}: cost, FDP, power', fontsize=TITLE_SIZE, pad=4)
    ax_top.set_ylim(0, nice_upper_bound(np.nanmax(cost), default=1.0))
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_top.set_ylabel('Cost')

    # FDR on right axis.
    ax_fdr = ax_top.twinx()
    ax_fdr.plot(x, fdr, color=FDR_COLOR, marker='o', linewidth=1.0, markersize=2.8, zorder=3)
    upper_fdr = max(0.10, nice_upper_bound(np.nanmax(fdr), default=0.10))
    ax_fdr.set_ylim(0, upper_fdr)
    ax_fdr.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_fdr.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.2f}'))
    ax_fdr.tick_params(axis='y', labelsize=TICK_LABEL_SIZE, colors=FDR_COLOR, length=2.0)
    ax_fdr.set_ylabel('FDP', color=FDR_COLOR)
    normalize_axis_text(ax_fdr)

    # Bottom: power bars, inverted as in the original script.
    ax_bot.bar(x, power, width=0.50, color=POWER_COLOR, alpha=0.85, zorder=2)
    pmax = max(0.05, nice_upper_bound(np.nanmax(power), default=1.0))
    ax_bot.set_ylim(0, pmax)
    ax_bot.invert_yaxis()
    ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_bot.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: '' if np.isclose(y, 0) else f'{y:g}'))
    ax_bot.set_ylabel('Power')

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax_bot.set_xlabel(eta_label if show_xlabel else '')

    style_ax(ax_top)
    style_ax(ax_bot)
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_bot.tick_params(axis='x', top=False)

    # Put the first-panel legend below the first split panel instead of
    # inside the plot area or below the whole figure.  For multi-row
    # figures, this is shown only for the last row so the legend is not
    # repeated for every dataset.
    if show_legend:
        legend = ax_bot.legend(
            handles=first_panel_legend_handles(),
            loc='upper center',
            bbox_to_anchor=(0.5, -1.18),
            ncol=3,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            columnspacing=1.00,
            handletextpad=0.45,
            borderpad=0.45,
        )
        normalize_legend_text(legend)

    return ax_top, ax_bot, ax_fdr


def first_panel_legend_handles():
    return [
        Line2D([0], [0], color=COST_COLOR, linewidth=7, alpha=0.85, label='Average cost'),
        Line2D([0], [0], color=POWER_COLOR, linewidth=7, alpha=0.85, label='Power'),
        Line2D([0], [0], color=FDR_COLOR, marker='o', linewidth=1.0, markersize=4, label='FDP'),
    ]


# ============================================================
# Panel 2: delta metrics relative to eta = 0
# ============================================================
def plot_delta_metrics(ax, dsub, eta_col, eta_label, dataset_name, show_xlabel=True, show_legend=False):
    base = get_base_row(dsub, eta_col)
    if base is None:
        draw_empty_panel(ax, f'{dataset_name}: delta metrics', r'No $\eta=0$')
        return

    eta = dsub[eta_col].to_numpy(dtype=float)
    # Use evenly spaced plotting positions.  The eta grid is nonuniform
    # (0, 0.25, 0.5, 1, 2, ..., 10), so plotting against the raw eta
    # values makes the first several tick labels overlap.
    x = np.arange(len(eta))

    cost = dsub['mean_average_cost'].to_numpy(dtype=float)
    power = dsub['mean_power'].to_numpy(dtype=float)
    fdr = dsub['mean_fdr'].to_numpy(dtype=float)

    cost0 = float(base['mean_average_cost'])
    power0 = float(base['mean_power'])
    fdr0 = float(base['mean_fdr'])

    if cost0 > 0:
        delta_cost_reduction_pct = 100.0 * (cost0 - cost) / cost0
    else:
        delta_cost_reduction_pct = np.zeros_like(cost)

    delta_power_pp = 100.0 * (power - power0)
    delta_fdr_pp = 100.0 * (fdr - fdr0)

    ax.axhline(0.0, color=ZERO_COLOR, linewidth=0.9, linestyle='--', alpha=0.8)
    ax.plot(x, delta_cost_reduction_pct, color=COST_COLOR, marker='o', linewidth=1.35, markersize=4.0, label='Cost reduction (%)')
    ax.plot(x, delta_power_pp, color=POWER_COLOR, marker='s', linewidth=1.35, markersize=4.0, label=r'$\Delta$Power (pp)')
    ax.plot(x, delta_fdr_pp, color=FDR_COLOR, marker='^', linewidth=1.35, markersize=4.0, label=r'$\Delta$FDP (pp)')

    ymin, ymax = nice_limit_from_values(np.r_[delta_cost_reduction_pct, delta_power_pp, delta_fdr_pp], default=5.0)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(f'{dataset_name}: delta metrics', fontsize=TITLE_SIZE, pad=4)
    ax.set_ylabel(r'Difference from $\eta=0$')
    ax.set_xlabel(eta_label if show_xlabel else '')
    set_eta_ticks(ax, x, eta)
    style_ax(ax)
    if show_legend:
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.32),
            ncol=3,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            columnspacing=0.85,
            handletextpad=0.45,
            borderpad=0.45,
        )
        normalize_legend_text(legend)


# ============================================================
# Panel 3: delta selected-region counts relative to eta = 0
# ============================================================
def plot_delta_selected_counts(ax, dsub, eta_col, eta_label, dataset_name, show_xlabel=True, show_legend=False):
    base = get_base_row(dsub, eta_col)
    if base is None:
        draw_empty_panel(ax, f'{dataset_name}: selected-count changes', r'No $\eta=0$')
        return

    eta = dsub[eta_col].to_numpy(dtype=float)
    # Use evenly spaced plotting positions for the nonuniform eta grid.
    x = np.arange(len(eta))

    fail = dsub['mean_n_selected_fail'].to_numpy(dtype=float)
    ind = dsub['mean_n_selected_ind'].to_numpy(dtype=float)
    passed = dsub['mean_n_selected_pass'].to_numpy(dtype=float)

    delta_fail = fail - float(base['mean_n_selected_fail'])
    delta_ind = ind - float(base['mean_n_selected_ind'])
    delta_pass = passed - float(base['mean_n_selected_pass'])

    ax.axhline(0.0, color=ZERO_COLOR, linewidth=0.9, linestyle='--', alpha=0.8)
    ax.plot(x, delta_fail, color=FAIL_COLOR, marker='o', linewidth=1.35, markersize=4.0, label=r'$\Delta n_{\mathrm{fail}}$')
    ax.plot(x, delta_ind, color=IND_COLOR, marker='s', linewidth=1.35, markersize=4.0, label=r'$\Delta n_{\mathrm{ind}}$')
    ax.plot(x, delta_pass, color=PASS_COLOR, marker='^', linewidth=1.35, markersize=4.0, label=r'$\Delta n_{\mathrm{pass}}$')

    ymin, ymax = nice_limit_from_values(np.r_[delta_fail, delta_ind, delta_pass], default=10.0)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(f'{dataset_name}: selected-count changes', fontsize=TITLE_SIZE, pad=4)
    ax.set_ylabel(r'Change in mean selected count')
    ax.set_xlabel(eta_label if show_xlabel else '')
    set_eta_ticks(ax, x, eta)
    style_ax(ax)
    if show_legend:
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.32),
            ncol=3,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            columnspacing=0.85,
            handletextpad=0.45,
            borderpad=0.45,
        )
        normalize_legend_text(legend)


# ============================================================
# Row figure
# ============================================================
def make_three_panel_rows_figure(dataset_list, sample, q_value, method, result_dir, out_dir, eta_grid, strict=False, file_format='pdf', dpi=300):
    nrows = len(dataset_list)
    ncols = 3

    if nrows <= 0:
        raise ValueError('At least one dataset must be provided.')

    # Large enough for 15 datasets, but not too large for a single dataset.
    # The first column contains a split panel, so a slightly taller row
    # prevents titles, x tick labels, and the global legend from colliding.
    row_height = 5.15
    fig_height = max(6.90, row_height * nrows)
    fig_width = 22.0
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.36, hspace=0.48)

    for row, dataset_name in enumerate(dataset_list):
        df = load_dataset_summary(result_dir, dataset_name, sample, strict=strict)
        dsub, eta_col, eta_label = get_method_q_data(df, method, q_value, eta_grid)

        if dsub.empty:
            plot_power_cost_hist_cell(fig, outer[row, 0], pd.DataFrame(), None, r'$\eta$', dataset_name, q_value, show_xlabel=True, show_legend=(row == nrows - 1))
            ax_delta = fig.add_subplot(outer[row, 1])
            ax_counts = fig.add_subplot(outer[row, 2])
            draw_empty_panel(ax_delta, f'{dataset_name}: delta metrics', 'No data')
            draw_empty_panel(ax_counts, f'{dataset_name}: selected-count changes', 'No data')
            continue

        plot_power_cost_hist_cell(
            fig, outer[row, 0], dsub, eta_col, eta_label, dataset_name, q_value,
            show_xlabel=True,
            show_legend=(row == nrows - 1)
        )

        ax_delta = fig.add_subplot(outer[row, 1])
        ax_counts = fig.add_subplot(outer[row, 2])
        plot_delta_metrics(
            ax_delta, dsub, eta_col, eta_label, dataset_name,
            show_xlabel=True,
            show_legend=(row == nrows - 1)
        )
        plot_delta_selected_counts(
            ax_counts, dsub, eta_col, eta_label, dataset_name,
            show_xlabel=True,
            show_legend=(row == nrows - 1)
        )

    method_tag = method_to_tag(method)
    q_tag = q_to_tag(q_value)
    dataset_tag = 'all' if len(dataset_list) == len(ALL_DATASETS) else '_'.join(dataset_list)
    dataset_tag = dataset_tag.replace('/', '-').replace(' ', '_')

    # Use margins measured in inches.  The figure-level title is intentionally
    # omitted for paper-style figures; the caption should carry the full title.
    # Only the three panel titles are retained inside the figure.
    top_margin_in = 1.05
    bottom_margin_in = 2.25
    top_frac = 1.0 - top_margin_in / fig_height
    bottom_frac = bottom_margin_in / fig_height

    fig.subplots_adjust(
        left=0.055,
        right=0.965,
        top=top_frac,
        bottom=bottom_frac,
        hspace=0.68,
        wspace=0.46,
    )

    filename = f'{method_tag}_{dataset_tag}_three_panel_rows_{q_tag}.{file_format}'
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    print(f'[Saved] {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=float, help='Sample ratio, e.g. 0.10 or 1.00')
    parser.add_argument('--result_dir', type=str, default='result-cost-0513',
                        help='Directory containing per-dataset aggregated_over_seeds.csv files')
    parser.add_argument('--output_dir', type=str, default='figure-cost-0518raw-three-panel',
                        help='Directory to save generated figures')
    parser.add_argument('--datasets', type=str, nargs='+', default=ALL_DATASETS,
                        help='Datasets to plot. Default: all 15 QSAR datasets')
    parser.add_argument('--q_values', type=float, nargs='+', default=[0.2],
                        help='Nominal FDP levels to plot. Default: 0.2')
    parser.add_argument('--methods', type=str, nargs='+', default=['RSI-EC', 'RSI-CS'],
                        choices=['RSI-EC', 'RSI-CS'],
                        help='Methods to plot. Default: RSI-EC RSI-CS')
    # parser.add_argument('--eta_grid', type=str,
    #                     default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
    #                     help='Comma-separated eta grid. Default: 0,0.25,0.5,1,2,...,10')
    parser.add_argument('--eta_grid', type=str,
                        default='0,0.25,0.5,1,2,3,4,5,6,7,8,9,10',
                        help='Comma-separated eta grid. Default: 0,0.25,0.5,1,2,...,10')
    parser.add_argument('--file_format', type=str, default='pdf', choices=['pdf', 'png', 'svg'],
                        help='Output format. Default: pdf')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures. Default: 300')
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
    print('[Info] file_format:', args.file_format)

    for q in args.q_values:
        for method in args.methods:
            make_three_panel_rows_figure(
                dataset_list=args.datasets,
                sample=sample,
                q_value=q,
                method=method,
                result_dir=args.result_dir,
                out_dir=out_dir,
                eta_grid=eta_grid,
                strict=args.strict,
                file_format=args.file_format,
                dpi=args.dpi,
            )

    print('All figures saved to:', out_dir)


if __name__ == '__main__':
    main()

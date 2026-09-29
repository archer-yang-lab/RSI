import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import argparse

# ---------- style ----------
BRIGHT_BLUE = '#4C9EF1'
BRIGHT_GREEN = '#5CBF60'
BRIGHT_RED = '#E85B5B'
BRIGHT_ORANGE = '#F5A623'
LIGHT_GRAY = '#D9D9D9'
NOMINAL_FDP_TICKS = [0,0.2,0.4,0.6,0.8,1.0]

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': '#666666',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'grid.color': LIGHT_GRAY,
    'grid.alpha': 0.8,
    'grid.linestyle': '-',
})

dataset_list = [
    '3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT',
    'LOGD', 'METAB', 'NK1', 'OX1', 'OX2',
    'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN'
]

parser = argparse.ArgumentParser()
parser.add_argument('n_itr', type=int)
args = parser.parse_args()
n_itr = args.n_itr

out_dir = 'figure-score09'
os.makedirs(out_dir, exist_ok=True)


def load_score_data(sample):
    df_list = []

    for name in dataset_list:
        df_ones = []
        for j in range(1, n_itr + 1):
            file_path = os.path.join(
                "result-score09",
                f"{name} {sample:.2f}",
                f"{name} {sample:.2f} {j}.csv"
            )
            df = pd.read_csv(file_path)
            df_ones.append(df)

        df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()
        df_list.append(df)

    return df_list


def prepare_figure():
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(22, 13.5))
    axs = axs.flatten()

    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', labelsize=18, width=1.2, length=5)

    return fig, axs


def style_spines(axs):
    for ax in axs:
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#666666')


def plot_panel(ax, x, y, label, color):
    ax.plot(
        x, y,
        label=label,
        marker='o',
        color=color,
        alpha=0.8,
        linewidth=2.5,
        markersize=5
    )


def draw_grid(
        df_list, x_cols, y_cols, xlabel, ylabel, filename,
        diag=False, xticks=None
):
    fig, axs = prepare_figure()

    for i, name in enumerate(dataset_list):
        ax = axs[i]
        df = df_list[i]

        plot_panel(ax, df[x_cols[0]], df[y_cols[0]], 'Signed error score', BRIGHT_BLUE)
        plot_panel(ax, df[x_cols[1]], df[y_cols[1]], 'Clipped score', BRIGHT_ORANGE)
        plot_panel(ax, df[x_cols[2]], df[y_cols[2]], 'Uncertainty signed error score', BRIGHT_GREEN)
        plot_panel(ax, df[x_cols[3]], df[y_cols[3]], 'Uncertainty clipped score', BRIGHT_RED)

        if diag:
            ax.plot([0.05, 0.55], [0.05, 0.55],
                    color='grey', alpha=0.7, linestyle='-.')

        if xticks is not None:
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.set_title(name, fontsize=20)

    style_spines(axs)

    fig.subplots_adjust(
        wspace=0.24,
        hspace=0.28,
        top=0.92,
        bottom=0.18,
        left=0.08,
        right=0.96
    )

    fig.text(0.5, 0.10, xlabel, ha='center', fontsize=24)
    fig.text(0.03, 0.5, ylabel, va='center', rotation='vertical', fontsize=24)

    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=True,
        fontsize=22,
        columnspacing=2.5,
        handletextpad=0.9,
        borderpad=0.6,
        labelspacing=0.8,
        handlelength=2.8
    )
    legend.get_frame().set_edgecolor('#999999')
    legend.get_frame().set_linewidth(1.2)

    base_name = os.path.splitext(filename)[0]
    plt.savefig(os.path.join(out_dir, f"{base_name}.pdf"), facecolor='white')
    plt.close(fig)


def run_for_sample(sample, suffix=''):
    df_list = load_score_data(sample)

    # FDP control
    draw_grid(
        df_list=df_list,
        x_cols=['fdp_nominals', 'fdp_nominals', 'fdp_nominals', 'fdp_nominals'],
        y_cols=['fdpn_cssigned', 'fdpn_cs', 'fdpn_csunsigned', 'fdpn_csun'],
        xlabel='Nominal FDP',
        ylabel='Observed FDP',
        filename=f'compfdpscore{suffix}.pdf',
        diag=True,
        xticks=NOMINAL_FDP_TICKS
    )

    # Observed FDP vs Observed Power
    draw_grid(
        df_list=df_list,
        x_cols=['fdpn_cssigned', 'fdpn_cs', 'fdpn_csunsigned', 'fdpn_csun'],
        y_cols=['powern_cssigned', 'powern_cs', 'powern_csunsigned', 'powern_csun'],
        xlabel='Observed FDP',
        ylabel='Observed Power',
        filename=f'comppowerobscore{suffix}.pdf',
        diag=False
    )

    # Nominal FDP vs Observed Power
    draw_grid(
        df_list=df_list,
        x_cols=['fdp_nominals', 'fdp_nominals', 'fdp_nominals', 'fdp_nominals'],
        y_cols=['powern_cssigned', 'powern_cs', 'powern_csunsigned', 'powern_csun'],
        xlabel='Nominal FDP',
        ylabel='Observed Power',
        filename=f'comppowerscore{suffix}.pdf',
        diag=False,
        xticks=NOMINAL_FDP_TICKS
    )


# sample = 1.0
run_for_sample(1.0, '')

# sample = 0.10
run_for_sample(0.10, '0.1')

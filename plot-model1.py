import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse

# ---------- style ----------
BRIGHT_BLUE = '#4C9EF1'
BRIGHT_GREEN = '#5CBF60'
BRIGHT_RED = '#E85B5B'
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

model_info = {
    'rf': ('Random Forest', BRIGHT_BLUE),
    'lin': ('Linear Regression', BRIGHT_RED),
    'nn': ('MLP', BRIGHT_GREEN),
}

parser = argparse.ArgumentParser()
parser.add_argument('n_itr', type=int)
args = parser.parse_args()
n_itr = args.n_itr

out_dir = 'figure-model'
os.makedirs(out_dir, exist_ok=True)


def empty_result_df():
    return pd.DataFrame(columns=[
        'fdp_nominals', 'fdpn_cs', 'powern_cs', 'fdpn_sh', 'powern_sh'
    ])


def load_one_model(sample, model):
    result_list = []

    for name in dataset_list:
        df_ones = []

        for j in range(1, n_itr + 1):
            file_path = os.path.join(
                "result-model",
                model,
                f"{name} {sample:.2f}",
                f"{name} {sample:.2f} {j}.csv"
            )

            try:
                df = pd.read_csv(file_path)
                df_ones.append(df)
            except FileNotFoundError as e:
                print(e)
                continue

        if len(df_ones) == 0:
            df = empty_result_df()
        else:
            df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

        result_list.append(df)

    return result_list


def prepare_figure():
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(22, 13.5))
    axs = axs.flatten()

    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', labelsize=18)

    return fig, axs


def style_spines(axs):
    for ax in axs:
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor('gray')


def plot_panel(ax, x, y, label, color):
    ax.plot(x, y, label=label, marker='o', color=color, alpha=0.8, linewidth=2.5, markersize=7)


def draw_grid(rf_list, lin_list, nn_list, x_col, y_col, xlabel, ylabel,
              filename, diag=False):
    fig, axs = prepare_figure()

    for i, name in enumerate(dataset_list):
        ax = axs[i]

        plot_panel(ax, rf_list[i][x_col], rf_list[i][y_col], f'Random Forest', BRIGHT_BLUE)
        plot_panel(ax, lin_list[i][x_col], lin_list[i][y_col], f'Linear Regression', BRIGHT_RED)
        plot_panel(ax, nn_list[i][x_col], nn_list[i][y_col], f'MLP', BRIGHT_GREEN)

        if diag:
            ax.plot([0.05, 0.55], [0.05, 0.55], color='grey', alpha=0.7, linestyle='-.')

        if x_col == 'fdp_nominals':
            ax.set_xlim(0, 1)
            ax.set_xticks(NOMINAL_FDP_TICKS)
            ax.set_xticklabels(['0'] + [f'{tick:.1f}' for tick in NOMINAL_FDP_TICKS[1:]])

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
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=True,
        fontsize=24,
        columnspacing=2.5,
        handletextpad=0.9,
        borderpad=0.6,
        labelspacing=0.8
    )

    plt.savefig(os.path.join(out_dir, filename), facecolor='white')
    plt.close(fig)


def run_for_sample(sample, suffix=''):
    nn_list = load_one_model(sample, 'nn')
    lin_list = load_one_model(sample, 'lin')
    rf_list = load_one_model(sample, 'rf')

    # comparison: CS
    draw_grid(
        rf_list, lin_list, nn_list,
        x_col='fdp_nominals',
        y_col='fdpn_cs',
        xlabel='Nominal FDP',
        ylabel='Observed FDP',
        filename=f'compfdpcs{suffix}.pdf',
        diag=True
    )

    draw_grid(
        rf_list, lin_list, nn_list,
        x_col='fdpn_cs',
        y_col='powern_cs',
        xlabel='Observed FDP',
        ylabel='Observed Power',
        filename=f'comppowerobcs{suffix}.pdf',
        diag=False
    )

    draw_grid(
        rf_list, lin_list, nn_list,
        x_col='fdp_nominals',
        y_col='powern_cs',
        xlabel='Nominal FDP',
        ylabel='Observed Power',
        filename=f'comppowercs{suffix}.pdf',
        diag=False
    )

    # sheridan method
    draw_grid(
        rf_list, lin_list, nn_list,
        x_col='fdp_nominals',
        y_col='fdpn_sh',
        xlabel='Nominal FDP',
        ylabel='Observed FDP',
        filename=f'compfdpsh{suffix}.pdf',
        diag=True
    )

    draw_grid(
        rf_list, lin_list, nn_list,
        x_col='fdp_nominals',
        y_col='powern_sh',
        xlabel='Nominal FDP',
        ylabel='Observed Power',
        filename=f'comppowersh{suffix}.pdf',
        diag=False
    )

    draw_grid(
        rf_list, lin_list, nn_list,
        x_col='fdpn_sh',
        y_col='powern_sh',
        xlabel='Observed FDP',
        ylabel='Observed Power',
        filename=f'comppowerobsh{suffix}.pdf',
        diag=False
    )


# sample = 1.0
run_for_sample(1.0, '')

# sample = 0.10
run_for_sample(0.10, '0.1')

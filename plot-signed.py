import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('sample', type=float)
parser.add_argument('n_itr', type=int)
args = parser.parse_args()

sample = 1.0
n_itr = args.n_itr

# Set ggplot style for the plots
plt.style.use('ggplot')

df_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']


for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        df = pd.read_csv(
            os.path.join("result-signedc1", f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()
    df_list.append(df)

out_dir = os.path.join('figure-signedc1')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

'''comparison'''

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(4.7, -3.0), frameon=True, shadow=False, ncol=2, fontsize=22)
    else:
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

    # Set axis labels
    ax.plot([0.05, 0.55], [0.05, 0.55], color='grey', alpha=0.7, linestyle='-.')
    ax.set_title(f'{name}', fontsize=22)

    # Add grid lines
    ax.grid(True)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.92, bottom=0.2, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.13, 'Nominal FDP', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed FDP', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
#fig.suptitle("Comparison: FDP control for all 15 Datasets", fontsize=16)

# Display the plot
# plt.savefig(os.path.join("figure-signedc1", "compfdp3_{model}.png"))
plt.savefig(os.path.join("figure-signedc1", "compfdpscore.png"))
# plt.show()


# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdpn_cssigned'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(df_list[i]['fdpn_cs'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(df_list[i]['fdpn_csunsigned'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        line4, = ax.plot(df_list[i]['fdpn_csun'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)
        ax.legend(loc='best', bbox_to_anchor=(4.7, -3.0), frameon=True, shadow=False, ncol=2, fontsize=22)
    else:
        ax.plot(df_list[i]['fdpn_cssigned'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)
        ax.plot(df_list[i]['fdpn_cs'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)
        ax.plot(df_list[i]['fdpn_csunsigned'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)
        ax.plot(df_list[i]['fdpn_csun'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

    # Set axis labels
    ax.set_title(f'{name}', fontsize=22)

    # Add grid lines
    ax.grid(True)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.92, bottom=0.2, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.13, 'Observed FDP', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Power', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
#fig.suptitle("Comparison: Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("figure-signedc1", "comppowerobscore.png"))
# plt.show()


# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(4.7, -3.0), frameon=True, shadow=False, ncol=2, fontsize=22)
    else:
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

    # Set axis labels
    ax.set_title(f'{name}', fontsize=22)

    # Add grid lines
    ax.grid(True)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.92, bottom=0.2, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.13, 'Nominal FDP', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Power', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
#fig.suptitle("Comparison: Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("figure-signedc1", "comppowerscore.png"))
# plt.show()

#######################0.1dataset########################
# Set ggplot style for the plots
parser = argparse.ArgumentParser()
# parser.add_argument('sample', type=float)
parser.add_argument('seed', type=int)
args = parser.parse_args()

sample = 0.10
n_itr = args.seed
plt.style.use('ggplot')

df_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        df = pd.read_csv(
            os.path.join("result-signedc1", f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()
    df_list.append(df)

out_dir = os.path.join('figure-signedc1')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

'''comparison'''

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(4.7, -3.0), frameon=True, shadow=False, ncol=2, fontsize=22)
    else:
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

    # Set axis labels
    ax.plot([0.05, 0.55], [0.05, 0.55], color='grey', alpha=0.7, linestyle='-.')
    ax.set_title(f'{name}', fontsize=22)

    # Add grid lines
    ax.grid(True)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.92, bottom=0.2, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.13, 'Nominal FDP', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed FDP', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
#fig.suptitle("Comparison: FDP control for all 15 Datasets", fontsize=16)

# Display the plot
# plt.savefig(os.path.join("figure-signedc1", "compfdp3_{model}.png"))
plt.savefig(os.path.join("figure-signedc1", "compfdp0.1score.png"))
# plt.show()


# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdpn_cssigned'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(df_list[i]['fdpn_cs'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(df_list[i]['fdpn_csunsigned'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        line4, = ax.plot(df_list[i]['fdpn_csun'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)
        ax.legend(loc='best', bbox_to_anchor=(4.7, -3.0), frameon=True, shadow=False, ncol=2, fontsize=22)
    else:
        ax.plot(df_list[i]['fdpn_cssigned'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)
        ax.plot(df_list[i]['fdpn_cs'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)
        ax.plot(df_list[i]['fdpn_csunsigned'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)
        ax.plot(df_list[i]['fdpn_csun'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

    # Set axis labels
    ax.set_title(f'{name}', fontsize=22)

    # Add grid lines
    ax.grid(True)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.92, bottom=0.2, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.13, 'Observed FDP', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Power', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
#fig.suptitle("Comparison: Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("figure-signedc1", "comppowerob0.1score.png"))
# plt.show()


# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)

        line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(4.7, -3.0), frameon=True, shadow=False, ncol=2, fontsize=22)
    else:
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cssigned'],
                         label='signed error score', marker='o', color='steelblue', alpha=0.8)
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_cs'],
                         label='clipped score', marker='o', color='orange', alpha=0.8)
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csunsigned'],
                         label='uncertainty signed error score', marker='o', color='darkgreen', alpha=0.8)
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['powern_csun'],
                         label='uncertainty clipped score', marker='o', color='red', alpha=0.8)

    # Set axis labels
    ax.set_title(f'{name}', fontsize=22)

    # Add grid lines
    ax.grid(True)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.92, bottom=0.2, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.13, 'Nominal FDP', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Power', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
#fig.suptitle("Comparison: Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("figure-signedc1", "comppower0.1score.png"))
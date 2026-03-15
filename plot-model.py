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

model = 'nn'

# Set ggplot style for the plots
plt.style.use('ggplot')

nn_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        try:
            df = pd.read_csv(os.path.join(f"result-model", model, f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

    # if to only 0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    nn_list.append(df)

model = 'lin'

# Set ggplot style for the plots
plt.style.use('ggplot')

lin_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        try:
            df = pd.read_csv(os.path.join(f"result-model", model, f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

    # if to only 0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    lin_list.append(df)

model = 'rf'

# Set ggplot style for the plots
plt.style.use('ggplot')

rf_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        try:
            df = pd.read_csv(os.path.join(f"result-model", model, f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

    # if to only 0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    rf_list.append(df)
out_dir = os.path.join('figure-model')
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
        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_cs'],
                         label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_cs'],
                         label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_cs'],
                         label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

        # line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_sh'],
        #                  label='Single-stage Sheridan: FDP', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.0, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_cs'],
                label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_cs'],
                label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_cs'],
                label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"compfdpcs.png"))
# plt.show()


# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdps_1new2'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='darkgreen', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdpn_cs'], rf_list[i]['powern_cs'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdpn_cs'], lin_list[i]['powern_cs'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdpn_cs'], nn_list[i]['powern_cs'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)
        # line4, = ax.plot(df_list[i]['fdpn_sh'], df_list[i]['powern_sh'],
        #                  label='Single-stage Sheridan: Power', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)
        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdpn_cs'], rf_list[i]['powern_cs'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdpn_cs'], lin_list[i]['powern_cs'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdpn_cs'], nn_list[i]['powern_cs'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowerobcs.png"))
# plt.show()

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdps_1new2'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='darkgreen', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_cs'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_cs'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_cs'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)
        # line4, = ax.plot(df_list[i]['fdpn_sh'], df_list[i]['powern_sh'],
        #                  label='Single-stage Sheridan: Power', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)
        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_cs'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_cs'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_cs'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowercs.png"))
# plt.show()

'''sheridan method'''
# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_sh'],
                         label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_sh'],
                         label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_sh'],
                         label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

        # line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_sh'],
        #                  label='Single-stage Sheridan: FDP', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.0, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_sh'],
                label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_sh'],
                label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_sh'],
                label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

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
# plt.savefig(os.path.join("figure-model", "compfdp3_{model}.png"))
plt.savefig(os.path.join("figure-model", f"compfdpsh.png"))
# plt.show()

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_sh'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_sh'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_sh'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_sh'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_sh'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_sh'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowersh.png"))
# plt.show()

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdpn_sh'], rf_list[i]['powern_sh'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdpn_sh'], lin_list[i]['powern_sh'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdpn_sh'], nn_list[i]['powern_sh'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdpn_sh'], rf_list[i]['powern_sh'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdpn_sh'], lin_list[i]['powern_sh'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdpn_sh'], nn_list[i]['powern_sh'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowerobsh.png"))
# plt.show()

#######################0.1dataset########################
# Set ggplot style for the plots
parser = argparse.ArgumentParser()
# parser.add_argument('sample', type=float)
parser.add_argument('seed', type=int)
args = parser.parse_args()

sample = 0.10
n_itr = args.seed
model = 'nn'

# Set ggplot style for the plots
plt.style.use('ggplot')

nn_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        try:
            df = pd.read_csv(os.path.join(f"result-model", model, f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

    # if to only 0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    nn_list.append(df)

model = 'lin'

# Set ggplot style for the plots
plt.style.use('ggplot')

lin_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        try:
            df = pd.read_csv(os.path.join(f"result-model", model, f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

    # if to only 0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    lin_list.append(df)

model = 'rf'

# Set ggplot style for the plots
plt.style.use('ggplot')

rf_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F',
                'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1 + n_itr):
        try:
            df = pd.read_csv(os.path.join(f"result-model", model, f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()

    # if to only 0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    rf_list.append(df)

out_dir = os.path.join('figure-model')

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
        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_cs'],
                         label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_cs'],
                         label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_cs'],
                         label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

        # line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_sh'],
        #                  label='Single-stage Sheridan: FDP', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.0, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_cs'],
                label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_cs'],
                label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_cs'],
                label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

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
# plt.savefig(os.path.join("figure-model", "compfdp3_{model}.png"))
plt.savefig(os.path.join("figure-model", f"compfdpcs0.1.png"))
# plt.show()


# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdps_1new2'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='darkgreen', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdpn_cs'], rf_list[i]['powern_cs'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdpn_cs'], lin_list[i]['powern_cs'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdpn_cs'], nn_list[i]['powern_cs'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)
        # line4, = ax.plot(df_list[i]['fdpn_sh'], df_list[i]['powern_sh'],
        #                  label='Single-stage Sheridan: Power', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)
        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdpn_cs'], rf_list[i]['powern_cs'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdpn_cs'], lin_list[i]['powern_cs'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdpn_cs'], nn_list[i]['powern_cs'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowerobcs0.1.png"))
# plt.show()

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdps_1new2'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='darkgreen', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_cs'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_cs'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_cs'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)
        # line4, = ax.plot(df_list[i]['fdpn_sh'], df_list[i]['powern_sh'],
        #                  label='Single-stage Sheridan: Power', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)
        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_cs'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_cs'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_cs'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowercs0.1.png"))
# plt.show()

'''sheridan method'''
# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_sh'],
                         label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_sh'],
                         label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_sh'],
                         label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

        # line4, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_sh'],
        #                  label='Single-stage Sheridan: FDP', marker='o', color='red', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.0, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['fdpn_sh'],
                label='Random Forest: FDP', marker='o', color='steelblue', alpha=0.8)

        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['fdpn_sh'],
                label='Linear Regression: FDP', marker='o', color='orange', alpha=0.8)

        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['fdpn_sh'],
                label='MLP: FDP', marker='o', color='darkgreen', alpha=0.8)

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
# plt.savefig(os.path.join("figure-model", "compfdp3_{model}.png"))
plt.savefig(os.path.join("figure-model", f"compfdpsh0.1.png"))
# plt.show()

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_sh'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_sh'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_sh'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdp_nominals'], rf_list[i]['powern_sh'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdp_nominals'], lin_list[i]['powern_sh'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdp_nominals'], nn_list[i]['powern_sh'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowersh0.1.png"))
# plt.show()

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()
# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #                  label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)

        line1, = ax.plot(rf_list[i]['fdpn_sh'], rf_list[i]['powern_sh'],
                         label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)

        line2, = ax.plot(lin_list[i]['fdpn_sh'], lin_list[i]['powern_sh'],
                         label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)

        line3, = ax.plot(nn_list[i]['fdpn_sh'], nn_list[i]['powern_sh'],
                         label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

        # line5, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdpn_cs'],
        #                  label='One-stage: FDP', marker='o', color='red', alpha=0.8)

        ax.legend(loc='best', bbox_to_anchor=(5.2, -3.0), frameon=True, shadow=False, ncol=3, fontsize=22)
    else:
        # ax.plot(df_list[i]['fdp_nominals'], df_list[i]['power1_new2'],
        #         label='Two-stage Sheridan: Power_1', marker='o', color='steelblue', alpha=0.8)
        ax.plot(rf_list[i]['fdpn_sh'], rf_list[i]['powern_sh'],
                label='Random Forest: Power', marker='o', color='steelblue', alpha=0.8)
        ax.plot(lin_list[i]['fdpn_sh'], lin_list[i]['powern_sh'],
                label='Linear Regression: Power', marker='o', color='orange', alpha=0.8)
        ax.plot(nn_list[i]['fdpn_sh'], nn_list[i]['powern_sh'],
                label='MLP: Power', marker='o', color='darkgreen', alpha=0.8)

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
plt.savefig(os.path.join("figure-model", f"comppowerobsh0.1.png"))
# plt.show()
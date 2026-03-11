from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import argparse
import time
import math
import sys

''' timer '''
class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time

def eval_inter(Y, rejected, lower, higher):
    true_reject = np.sum((Y < higher)&(Y > lower))
    # If rejected is empty, all outputs set to 0 or np.nan as fallback
    if len(rejected) == 0:
        fdp = 0.0
        power = 0.0
    else:
        fdp = np.sum((lower>= Y[rejected])|(Y[rejected]>= higher)) / len(rejected)
        power = np.sum((lower< Y[rejected]) & (Y[rejected]< higher)) / true_reject if true_reject > 0 else 0.0

    return fdp, power

def BH(calib_scores, test_scores, q=0.1, extra_info=None):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)

    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (
                    np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib + 1)

    # BH(q)
    df_test = pd.DataFrame({"id": range(ntest), "score": test_scores, "pval": pvals}).sort_values(by='pval')

    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j, 2] <= df_test.iloc[j, 3]]

    if len(idx_smaller) == 0:
        if not extra_info:
            return (np.array([]))
        elif extra_info == 'pval':
            return np.array([]), pvals
        else:
            return np.array([]), df_test
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        if not extra_info:
            return (idx_sel)
        elif extra_info == 'pval':
            return idx_sel, pvals
        else:
            return idx_sel, df_test

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('sample', type=float)
parser.add_argument('seed', type=int)
args = parser.parse_args()
random.seed(args.seed)

dataset_name = args.dataset
dataset_path = os.path.join('data', f'{dataset_name}_training_disguised.csv')

dataset = pd.read_csv(dataset_path)

if args.sample < 1:
    dataset = dataset.sample(frac=args.sample)

##
thresholds_map1 = {'NK1': 8.5, 'PGP': 0.1, 'LOGD': 3, '3A4': 4.5, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 7, 'METAB': 40, 'OX1': 5.8, 'OX2': 6, 'PPB': 1, 'RAT_F': 1.0, 'TDI': 0, 'THROMBIN': 6}
thresholds_map2 = {'NK1': 9.5, 'PGP': 0.8, 'LOGD': 5, '3A4': 6, 'CB1': 8.0, 'DPP4': 7, 'HIVINT': 7, 'HIVPROT': 9, 'METAB': 70, 'OX1': 7.5, 'OX2': 8, 'PPB': 2, 'RAT_F': 1.9, 'TDI': 1, 'THROMBIN': 9}


threshold_1 = thresholds_map1[dataset_name]
threshold_2 = thresholds_map2[dataset_name]

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=15/100, shuffle=True) # tc: train and calib

# ofdp_nominals = np.linspace(0.1, 0.5, 9)
fdp_nominals = np.linspace(0.1, 1.0, 9)
all_res = pd.DataFrame()
epsilon = 1e-8

# all_res['ofdp_nominals'] = ofdp_nominals
all_res['fdp_nominals'] = fdp_nominals

''' single stage'''

''' single stage sheridan '''
fdpn_sh, costn_sh, powern_sh, powers_sh= [], [], [], []
with Timer() as timer:
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rf_rmse = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50 / 70, shuffle=True)
    rf.fit(Xtrain, Ytrain)

    Ytrain_rmse_pred = rf.predict(Xtrain_rmse)
    all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in rf.estimators_])
    var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)

    rf_rmse.fit(np.column_stack((Ytrain_rmse_pred, var_train_rmse)), np.abs(Ytrain_rmse - Ytrain_rmse_pred))

    Ypred_calib1 = rf.predict(Xcalib1)
    all_Ypred_calib1 = np.column_stack([tree.predict(Xcalib1) for tree in rf.estimators_])
    var_calib1 = np.var(all_Ypred_calib1, axis=1)
    rmse_calib1 = rf_rmse.predict(np.column_stack((Ypred_calib1, var_calib1)))

    z_calib1_1 = (np.minimum(threshold_2 - Ypred_calib1, Ypred_calib1 - threshold_1)) / (rmse_calib1 + epsilon)
    # z_calib1_2 = (Ypred_calib1 - threshold_2) / rmse_calib1
    # z_calib1_1 = (np.minimum(threshold_1 - Ypred_calib1, Ypred_calib1 - threshold_2)) / (rmse_calib1 + epsilon)

    Ypred_test = rf.predict(Xtest)
    all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
    z_test_1 = (np.minimum(threshold_2 - Ypred_test,Ypred_test - threshold_1 )) / (rmse_test + epsilon)
    # z_test_1 = (np.minimum(threshold_1 - Ypred_test, Ypred_test - threshold_2)) / (rmse_test + epsilon)
    # z_test_2 = (Ypred_test - threshold_2) / rmse_test
    for i, fdp_nominal in enumerate(fdp_nominals):
        # 1. Reset min_R_1 at the START of each loop for fdp_nominal.
        # This is a critical bug fix from your original code.
        min_R_1 = None

        # Search for the optimal R on the calibration set
        for R in np.linspace(3, -3, 500):
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
            try_fdr, _ = eval_inter(Ycalib1, try_r_sel1, threshold_1, threshold_2)
            # Check if the current FDP meets the condition
            if np.any(fdp_nominal >= try_fdr):
                # Since R is decreasing, the last R that satisfies the condition is the minimum.
                min_R_1 = R

        if min_R_1 is None:
            # This ensures we reject nothing, leading to an FDP of 0.
            min_R_1 = np.inf
        # Because min_R_1 is now guaranteed to have a value, we can proceed directly.
        Sel_1 = [j for j in range(len(z_test_1)) if z_test_1[j] >= min_R_1]
        fdp, power = eval_inter(Ytest, Sel_1, threshold_1, threshold_2)

        fdpn_sh.append(fdp)
        powern_sh.append(power)

all_res['fdpn_sh'] = fdpn_sh
all_res['powern_sh'] = powern_sh
all_res['time_sh'] = [timer.runtime] * len(fdp_nominals)

''' single stage conformal selection '''
fdpn_cs, costn_cs, powern_cs, powers_cs= [], [], [], []
from sklearn.ensemble import RandomForestClassifier
with Timer() as timer:
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt',criterion="entropy")
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    rf.fit(Xtrain, ((threshold_1 < Ytrain) & (threshold_2 > Ytrain)))
    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000*((threshold_1 < Ycalib) & (threshold_2 > Ycalib)) - rf.predict_proba(Xcalib)[:, 1]  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -rf.predict_proba(Xtest)[:, 1]
        # print(test_scores[1:3])
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        # print(BH_2clip)
        fdp, power = eval_inter(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs.append(fdp)
        powern_cs.append(power)

all_res['fdpn_cs'] = fdpn_cs
all_res['powern_cs'] = powern_cs
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)

''' two stage conformal selection'''
fdpn_cs2inter, pcern_cs2inter, powern_cs2inter= [], [], []
with Timer() as timer:
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rf.fit(Xtrain, Ytrain)
    Ycalib_pred = rf.predict(Xcalib)
    Ytest_pred = rf.predict(Xtest)
    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000 * (threshold_1 <= Ycalib) - Ycalib_pred
        test_scores = -Ytest_pred
        BH_2clipstep1 = BH(calib_scores_2clip, test_scores, fdp_nominal)

        calib_scores_2clip = 1000 * (threshold_2 > Ycalib) - Ycalib_pred
        test_scores = -Ytest_pred
        BH_2clipstep2 = BH(calib_scores_2clip, test_scores, fdp_nominal)
        BH_2clip_inter = np.intersect1d(BH_2clipstep1, BH_2clipstep2)
        BH_2clip = BH_2clip_inter.astype(int)

        fdp, pcer, power = eval_inter(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs2inter.append(fdp)
        pcern_cs2inter.append(pcer)
        powern_cs2inter.append(power)


all_res['fdpn_cs2inter'] = fdpn_cs2inter
all_res['pcern_cs2inter'] = pcern_cs2inter
all_res['powern_cs2inter'] = powern_cs2inter


out_dir = os.path.join('result-inter', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-inter', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

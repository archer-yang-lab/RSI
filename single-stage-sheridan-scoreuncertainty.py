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
#here we add a cost in eval_o function
gray_cost=1
red_green_cost=2
def eval_new(Y, rejected_1, rejected_2, lower, higher):
    # true positives for each threshold
    true_reject_1 = np.sum(Y <= lower)
    true_reject_2 = np.sum(Y >= higher)

    # If rejected_1 is empty, all outputs set to 0 or np.nan as fallback
    if len(rejected_1) == 0:
        fdp_1 = 0.0
        fdp_2 = 0.0
        power1 = 0.0
        power2 = 0.0
    else:
        # safe index
        Y_r1 = Y[rejected_1]
        if len(rejected_2) == 0:
            fdp_1 = np.sum(lower < Y_r1) / len(rejected_1)
            fdp_2 = np.sum((lower < Y_r1) & (higher > Y_r1)) / len(rejected_1)
            power1 = np.sum(lower >= Y_r1) / true_reject_1 if true_reject_1 > 0 else 0.0
            power2 = np.sum((lower >= Y_r1) | (higher <= Y_r1)) / (true_reject_1 + true_reject_2) if (true_reject_1 + true_reject_2) > 0 else 0.0
        else:
            union12 = np.array(list(set(rejected_1) | set(rejected_2)))
            Y_union = Y[union12]
            fdp_1 = np.sum(lower < Y[rejected_1]) / len(rejected_1)
            fdp_2 = np.sum((lower < Y_union) & (higher > Y_union)) / len(union12)
            power1 = np.sum(lower >= Y[rejected_1]) / true_reject_1 if true_reject_1 > 0 else 0.0
            power2 = np.sum((lower >= Y_union) | (higher <= Y_union)) / (true_reject_1 + true_reject_2) if (true_reject_1 + true_reject_2) > 0 else 0.0

    return fdp_1, fdp_2, power1, power2

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

# thresholds_map1 = {'NK1': 6.5, 'PGP': -0.3, 'LOGD': 1.5, '3A4': 4.35, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 4.5, 'METAB': 40, 'OX1': 5, 'OX2': 6, 'PPB': 1, 'RAT_F': 0.3, 'TDI': 0, 'THROMBIN': 6}
# thresholds_map2 = {'NK1': 8.5, 'PGP': 0.5, 'LOGD': 3, '3A4': 6, 'CB1': 8.0, 'DPP4': 7, 'HIVINT': 7, 'HIVPROT': 6.5, 'METAB': 60, 'OX1': 6.5, 'OX2': 8, 'PPB': 2, 'RAT_F': 1.5, 'TDI': 1, 'THROMBIN': 7}
# # thresholds_map = {'NK1': 9.5, 'PGP': 0.5, 'LOGD': 4, '3A4': 4.5, 'CB1': 8.0, 'DPP4': 6.5, 'HIVINT': 7, 'HIVPROT': 9, 'METAB': 70, 'OX1': 7.5, 'OX2': 8, 'PPB': 2, 'RAT_F': 1.7, 'TDI': 1, 'THROMBIN': 7}

# Threshold mappings
# thresholds_map1 = {'NK1': 9.75, 'PGP': 0.92, 'LOGD': 4.0, '3A4': 4.35, 'CB1': 7.83, 'DPP4': 6.85, 'HIVINT': 6.7, 'HIVPROT': 9.76, 'METAB': 51, 'OX1': 7.28, 'OX2': 8.53, 'PPB': 2.3, 'RAT_F': 1.92, 'TDI': 0.67, 'THROMBIN': 7.76}
# thresholds_map2 = {'NK1': 10, 'PGP': 1.5, 'LOGD': 5, '3A4': 6.5, 'CB1': 9.0, 'DPP4': 7, 'HIVINT': 7.5, 'HIVPROT': 10.5, 'METAB': 70, 'OX1': 9, 'OX2': 10, 'PPB': 3.0, 'RAT_F': 2.3, 'TDI': 1.2, 'THROMBIN': 9}
# thresholds_map = {'NK1': 8.5, 'PGP': 0.5, 'LOGD': 3, '3A4': 4.5, 'CB1': 7.0, 'DPP4': 6.5, 'HIVINT': 6.5, 'HIVPROT': 6.5, 'METAB': 60, 'OX1': 6.0, 'OX2': 7, 'PPB': 1.5, 'RAT_F': 1.0, 'TDI': 0.5, 'THROMBIN': 6.5}

##
thresholds_map1 = {'NK1': 8.5, 'PGP': 0.1, 'LOGD': 3, '3A4': 4.5, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 7, 'METAB': 40, 'OX1': 5.8, 'OX2': 6, 'PPB': 1, 'RAT_F': 1.0, 'TDI': 0, 'THROMBIN': 6}
thresholds_map2 = {'NK1': 9.5, 'PGP': 0.8, 'LOGD': 5, '3A4': 6, 'CB1': 8.0, 'DPP4': 7, 'HIVINT': 7, 'HIVPROT': 9, 'METAB': 70, 'OX1': 7.5, 'OX2': 8, 'PPB': 2, 'RAT_F': 1.9, 'TDI': 1, 'THROMBIN': 9}


threshold_1 = thresholds_map1[dataset_name]
threshold_2 = thresholds_map2[dataset_name]
# threshold_0 = thresholds_map[dataset_name]

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=15/100, shuffle=True) # tc: train and calib

# ofdp_nominals = np.linspace(0.1, 0.5, 9)
fdp_nominals = np.linspace(0.1, 1.0, 9)
all_res = pd.DataFrame()
epsilon = 1e-8

# all_res['ofdp_nominals'] = ofdp_nominals
all_res['fdp_nominals'] = fdp_nominals

''' new two sheridan method '''
fdps_1new2, fdps_2new2, power1_new2, power2_new2 = [],[],[],[]
# the first step
# 15% test, 15% calibration2, 50% train, 20% calibration1
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

    z_calib1_1 = (threshold_1 - Ypred_calib1) / (rmse_calib1 + epsilon)
    # z_calib1_2 = (np.maximum(threshold_1 - Ypred_calib1, Ypred_calib1 - threshold_2)) / rmse_calib1
    z_calib1_2 = (Ypred_calib1 - threshold_2) / (rmse_calib1 + epsilon)

    Ypred_test = rf.predict(Xtest)
    all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
    z_test_1 = (threshold_1 - Ypred_test) / (rmse_test + epsilon)
    z_test_2 = (Ypred_test - threshold_2) / (rmse_test + epsilon)
    min_R_1 = None  # Initialize as None
    min_R_2 = None

    for fdp_nominal in fdp_nominals:
        # 1. Reset variables at the start of the loop for a clean slate on each iteration.
        min_R_1 = None
        min_R_2 = None

        # --- Stage 1: Find min_R_1 ---
        for R in np.linspace(3, -3, 500):
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
            try_fdr1, _, _, _ = eval_new(Ycalib1, try_r_sel1, [], threshold_1, threshold_2)
            if np.any(fdp_nominal >= try_fdr1):
                # Since R is decreasing, the last value that satisfies the condition will be the minimum.
                min_R_1 = R

        # 2. Add fallback logic for min_R_1.
        # If the loop above failed to find a suitable R, use the most conservative threshold.
        if min_R_1 is None:
            min_R_1 = np.inf

        # --- Stage 2: Find min_R_2 ---
        # Since min_R_1 is now guaranteed to have a value, we can proceed to find min_R_2.
        for R in np.linspace(3, -3, 500):
            # Note: try_r_sel1 here should use the already-determined min_R_1.
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= min_R_1]
            try_r_sel2 = [j for j in range(len(z_calib1_2)) if z_calib1_2[j] >= R]
            _, try_fdr2, _, _ = eval_new(Ycalib1, try_r_sel1, try_r_sel2, threshold_1, threshold_2)
            if np.any(fdp_nominal >= try_fdr2):
                min_R_2 = R

        # 3. Add fallback logic for min_R_2.
        if min_R_2 is None:
            min_R_2 = np.inf

        # --- Stage 3: Calculate results on the test set ---
        # 4. Directly execute the calculation, no more `if` checks needed.
        # Because min_R_1 and min_R_2 are now guaranteed to have values.
        Sel_1 = [j for j in range(len(z_test_1)) if z_test_1[j] >= min_R_1]
        Sel_2 = [j for j in range(len(z_test_2)) if z_test_2[j] >= min_R_2]

        fdp_1, fdp_2, power1, power2 = eval_new(Ytest, Sel_1, Sel_2, threshold_1, threshold_2)

        # --- Stage 4: Save the results ---
        # The variables here are now always defined, so it's safe to append.
        fdps_1new2.append(fdp_1)
        fdps_2new2.append(fdp_2)
        power1_new2.append(power1)
        power2_new2.append(power2)


all_res['fdps_1new2'] = fdps_1new2
all_res['fdps_2new2'] = fdps_2new2
all_res['power1_new2'] = power1_new2
all_res['power2_new2'] = power2_new2
all_res['time_2'] = [timer.runtime] * len(fdp_nominals)

''' single stage'''
def eval_n(Y, rejected_1, lower, higher):
    true_reject_1 = np.sum((lower >= Y)|(Y >= higher))
    if len(rejected_1) == 0:
        fdp = 0
        power1 = 0
        cost = 0
        power = 0
    else:
        fdp = np.sum((lower < Y[rejected_1]) & (higher > Y[rejected_1])) / len(rejected_1)
        power1 = np.sum((lower >= Y[rejected_1])|(higher <= Y[rejected_1])) / true_reject_1 if true_reject_1 != 0 else 0
        set_1c = np.delete(Y, rejected_1)
        cost = gray_cost*len(set_1c)
        power = power1
    return fdp, power1, cost, power

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

    z_calib1_1 = (np.maximum(threshold_1 - Ypred_calib1, Ypred_calib1 - threshold_2)) / (rmse_calib1 + epsilon)
    # z_calib1_2 = (Ypred_calib1 - threshold_2) / rmse_calib1

    Ypred_test = rf.predict(Xtest)
    all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
    z_test_1 = (np.maximum(threshold_1 - Ypred_test,Ypred_test - threshold_2 )) / (rmse_test + epsilon)
    # z_test_2 = (Ypred_test - threshold_2) / rmse_test
    for i, fdp_nominal in enumerate(fdp_nominals):
        # 1. Reset min_R_1 at the START of each loop for fdp_nominal.
        # This is a critical bug fix from your original code.
        min_R_1 = None

        # Search for the optimal R on the calibration set
        for R in np.linspace(3, -3, 500):
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
            try_fdr, _, _, _ = eval_n(Ycalib1, try_r_sel1, threshold_1, threshold_2)
            # Check if the current FDP meets the condition
            if np.any(fdp_nominal >= try_fdr):
                # Since R is decreasing, the last R that satisfies the condition is the minimum.
                min_R_1 = R

        if min_R_1 is None:
            # This ensures we reject nothing, leading to an FDP of 0.
            min_R_1 = np.inf
        # Because min_R_1 is now guaranteed to have a value, we can proceed directly.
        Sel_1 = [j for j in range(len(z_test_1)) if z_test_1[j] >= min_R_1]
        fdp, power, cost, powers = eval_n(Ytest, Sel_1, threshold_1, threshold_2)

        fdpn_sh.append(fdp)
        powern_sh.append(power)
        costn_sh.append(cost)
        powers_sh.append(powers)

all_res['fdpn_sh'] = fdpn_sh
all_res['costn_sh'] = costn_sh
all_res['powern_sh'] = powern_sh
all_res['powers_sh'] = powers_sh
all_res['time_sh'] = [timer.runtime] * len(fdp_nominals)

''' single stage conformal selection '''
fdpn_cs, costn_cs, powern_cs, powers_cs= [], [], [], []
from sklearn.ensemble import RandomForestClassifier
with Timer() as timer:
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt',criterion="entropy")
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    rf.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))
    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000*((threshold_1 >= Ycalib) | (threshold_2 <= Ycalib)) - rf.predict_proba(Xcalib)[:, 1]  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -rf.predict_proba(Xtest)[:, 1]
        # print(test_scores[1:3])
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        # print(BH_2clip)
        fdp, power, cost,powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs.append(fdp)
        powern_cs.append(power)
        costn_cs.append(cost)
        powers_cs.append(powers)

all_res['fdpn_cs'] = fdpn_cs
all_res['costn_cs'] = costn_cs
all_res['powern_cs'] = powern_cs
all_res['powers_cs'] = powers_cs
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)

''' single stage conformal selection + uncertainty'''
fdpn_csun, costn_csun, powern_csun, powers_csun= [], [], [], []
from sklearn.ensemble import RandomForestClassifier
with Timer() as timer:
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', criterion="entropy")
    rf_rmse = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50 / 70, shuffle=True)
    rf.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))

    Ytrain_rmse_pred = rf.predict_proba(Xtrain_rmse)[:, 1]
    all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in rf.estimators_])
    var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)
    rf_rmse.fit(np.column_stack((Ytrain_rmse_pred, var_train_rmse)), np.abs(((threshold_1 >= Ytrain_rmse) | (threshold_2 <= Ytrain_rmse)) - Ytrain_rmse_pred))

    Ypred_calib1 = rf.predict_proba(Xcalib1)[:, 1]
    all_Ypred_calib1 = np.column_stack([tree.predict(Xcalib1) for tree in rf.estimators_])
    var_calib1 = np.var(all_Ypred_calib1, axis=1)
    rmse_calib1 = rf_rmse.predict(np.column_stack((Ypred_calib1, var_calib1)))

    Ypred_test = rf.predict_proba(Xtest)[:, 1]
    all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = (1000*((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypred_calib1)/(rmse_calib1  + epsilon) # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -Ypred_test/(rmse_test + epsilon)
        # print(rmse_test[1:3])
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_csun.append(fdp)
        powern_csun.append(power)
        costn_csun.append(cost)
        powers_csun.append(powers)

all_res['fdpn_csun'] = fdpn_csun
all_res['costn_csun'] = costn_csun
all_res['powern_csun'] = powern_csun
all_res['powers_csun'] = powers_csun
all_res['time_csun'] = [timer.runtime] * len(fdp_nominals)

out_dir = os.path.join('result-uncertainty1.0', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-uncertainty1.0', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

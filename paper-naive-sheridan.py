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
    true_discovery = np.sum((lower <= Y) & (Y <= higher))
    if len(rejected) == 0:
        ofdp = 0
        opower = 0
    else:
        ofdp = np.sum((lower > Y[rejected])|(higher < Y[rejected])) / len(rejected)
        opower = np.sum((lower <= Y[rejected]) & (higher >= Y[rejected])) / true_discovery if true_discovery != 0 else 0
    return ofdp, opower

def eval_union(Y, rejected, lower, higher):
    true_discovery = np.sum((lower > Y) |( Y > higher))
    if len(rejected) == 0:
        ofdp = 0
        opower = 0
    else:
        ofdp = np.sum((lower <= Y[rejected]) & (higher >= Y[rejected])) / len(rejected)
        opower = np.sum((lower > Y[rejected])|(higher < Y[rejected]))  / true_discovery if true_discovery != 0 else 0
    return ofdp, opower

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

thresholds_map1 = {'NK1': 8.5, 'PGP': 0.1, 'LOGD': 3, '3A4': 4.5, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 7, 'METAB': 40, 'OX1': 5.8, 'OX2': 6, 'PPB': 1, 'RAT_F': 1.0, 'TDI': 0, 'THROMBIN': 6}
thresholds_map2 = {'NK1': 9.5, 'PGP': 0.8, 'LOGD': 5, '3A4': 6, 'CB1': 8.0, 'DPP4': 7, 'HIVINT': 7, 'HIVPROT': 9, 'METAB': 70, 'OX1': 7.5, 'OX2': 8, 'PPB': 2, 'RAT_F': 1.9, 'TDI': 1, 'THROMBIN': 9}


threshold_1 = thresholds_map1[dataset_name]
threshold_2 = thresholds_map2[dataset_name]

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=15/100, shuffle=True) # tc: train and calib

# fdp_nominals = np.linspace(0.1, 0.5, 9)
fdp_nominals = np.linspace(0.1, 1.0, 9)
all_res = pd.DataFrame()

all_res['fdp_nominals'] = fdp_nominals

fdpunion_2, fdpinter_2, powerunion_2, powerinter_2 = [],[],[],[]
''' two sheridan scores method '''
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

    #union of sets
    z_calib1_1 = (threshold_1 - Ypred_calib1) / rmse_calib1
    z_calib1_2 = (Ypred_calib1 - threshold_2) / rmse_calib1

    Ypred_test = rf.predict(Xtest)
    all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
    z_test_1 = (threshold_1 - Ypred_test) / rmse_test
    z_test_2 = (Ypred_test - threshold_2) / rmse_test

    for ofdp_nominal in fdp_nominals:  # Iterate through each value in ofdr_nominals
        min_R_1 = None  # Initialize as None
        min_R_2 = None
        # print(len(Set_1_cali2))
        for R in np.linspace(3,-3, 300):
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
            # print(try_r_sel1)
            try_ofdr, _ = eval_union(Ycalib1, try_r_sel1, threshold_1, threshold_2)
            # Check if the current try_ofdr meets the condition
            if ofdp_nominal >= try_ofdr:
                if min_R_1 is None or R < min_R_1:
                    min_R_1 = R
        # Only perform further operations if a valid min_R was found
        if min_R_1 is None:
            min_R_1 = np.inf
        if min_R_1 is not None:
            for R in np.linspace(3,-3,300):
                # try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= min_R_1]
                try_r_sel2 = [j for j in range(len(z_calib1_2)) if z_calib1_2[j] >= R]
                try_ofdr, _= eval_union(Ycalib1, try_r_sel2, threshold_1, threshold_2)
                # Check if the current try_ofdr meets the condition
                if ofdp_nominal >= try_ofdr:
                    if min_R_2 is None or R < min_R_2:
                        min_R_2 = R

            # print(min_R_1, min_R_2)
        if min_R_2 is None:
            min_R_2 = np.inf
        if min_R_1 is not None and min_R_2 is not None:
            Sel_1 = [j for j in range(len(z_test_1)) if z_test_1[j] >= min_R_1]
            Sel_2 = [j for j in range(len(z_test_2)) if z_test_2[j] >= min_R_2]
            set_1 = set(Sel_1)
            # print(len(Sel_1))
            set_2 = set(Sel_2)
            union_set = set_1 | set_2
            union = list(union_set)
            # print(union)

            # print(len(Sel_1), len(Sel_2))
            # print(BH_2clip,z_test,sheridan_15)
            ofdp, opower = eval_union(Ytest, union, threshold_1, threshold_2)
            # Print the results
            # print(ofdp)
            # print(f"Corresponding R_min_1 value: {min_R_1:.4f}, Corresponding R_min_2 value: {min_R_2:.4f}, oFDP nominal value: {ofdp_nominal:.4f}, Test oFDP value: {ofdp:.4f}, Test first-stage power value: {opower:.4f}")

        fdpunion_2.append(ofdp)
        powerunion_2.append(opower)

    #inter of sets
    z_calib1_1 = (Ypred_calib1 - threshold_1) / rmse_calib1
    z_calib1_2 = (threshold_2 - Ypred_calib1) / rmse_calib1

    Ypred_test = rf.predict(Xtest)
    all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
    z_test_2 = (threshold_2 - Ypred_test) / rmse_test
    z_test_1 = (Ypred_test - threshold_1) / rmse_test

    for ofdp_nominal in fdp_nominals:  # Iterate through each value in ofdr_nominals
        min_R_1 = None  # Initialize as None
        min_R_2 = None
        # print(len(Set_1_cali2))
        for R in np.linspace(3,-3, 300):
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
            # print(try_r_sel1)
            try_ofdr, _ = eval_inter(Ycalib1, try_r_sel1, threshold_1, threshold_2)
            # print(try_ofdr)
            # Check if the current try_ofdr meets the condition
            if ofdp_nominal >= try_ofdr:
                if min_R_1 is None or R < min_R_1:
                    min_R_1 = R
        if min_R_1 is None:
            min_R_1 = np.inf
        # Only perform further operations if a valid min_R was found
        if min_R_1 is not None:
            for R in np.linspace(3,-3,300):
                # try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= min_R_1]
                try_r_sel2 = [j for j in range(len(z_calib1_2)) if z_calib1_2[j] >= R]
                try_ofdr, _= eval_inter(Ycalib1, try_r_sel2, threshold_1, threshold_2)
                # Check if the current try_ofdr meets the condition
                if ofdp_nominal >= try_ofdr:
                    if min_R_2 is None or R < min_R_2:
                        min_R_2 = R
        if min_R_2 is None:
            min_R_2 = np.inf
        if min_R_1 is not None and min_R_2 is not None:
            Sel_1 = [j for j in range(len(z_test_1)) if z_test_1[j] >= min_R_1]
            Sel_2 = [j for j in range(len(z_test_2)) if z_test_2[j] >= min_R_2]
            set_1 = set(Sel_1)
            set_2 = set(Sel_2)
            intersection_set = set_1 & set_2
            intersection = list(intersection_set)

            # print(len(Sel_1), len(Sel_2))
            # print(BH_2clip,z_test,sheridan_15)
            ofdp, opower = eval_inter(Ytest, intersection, threshold_1, threshold_2)
            # Print the results
            # print(ofdp)
            # print(f"Corresponding R_min_1 value: {min_R_1:.4f}, Corresponding R_min_2 value: {min_R_2:.4f}, oFDP nominal value: {ofdp_nominal:.4f}, Test oFDP value: {ofdp:.4f}, Test first-stage power value: {opower:.4f}")

        fdpinter_2.append(ofdp)
        powerinter_2.append(opower)

        # power1_2.append(power1)
        # power2_2.append(power2)


all_res['fdp_nominals'] = fdp_nominals
all_res['fdpunion_2'] = fdpunion_2
all_res['powerunion_2'] = powerunion_2
all_res['fdpinter_2'] = fdpinter_2
all_res['powerinter_2'] = powerinter_2
all_res['time_2'] = [timer.runtime] * len(fdp_nominals)


out_dir = os.path.join('result-naive-sheridan', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-naive-sheridan', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

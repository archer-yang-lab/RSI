from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import argparse
import time

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

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str) # the name of the dataset to use
parser.add_argument('sample', type=float) # percentage of the dataset to consider
parser.add_argument('seed', type=int)
parser.add_argument('model', type=str, default='rf', choices=['rf', 'lin', 'nn']) # prediction model used
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# def get_model(mdl_str):
#     if mdl_str == 'rf':
#         return RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
#     if mdl_str == 'lin':
#         return Ridge(alpha=1e-5)
#     if mdl_str == 'nn':
#         return MLPRegressor(hidden_layer_sizes=[64, 64], max_iter=1000)
def get_model(mdl_str, random_state=0):
    if mdl_str == "rf":
        return RandomForestRegressor(
            n_estimators=100, max_depth=20, max_features="sqrt",
            random_state=random_state
        )
    if mdl_str == "lin":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1e-5))
        ])
    if mdl_str == "nn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=random_state
            ))
        ])
    raise ValueError("Unknown model")

dataset_name = args.dataset
dataset_path = os.path.join('data', f'{dataset_name}_training_disguised.csv')

dataset = pd.read_csv(dataset_path)

assert 0 < args.sample and args.sample <= 1
if args.sample < 1:
    dataset = dataset.sample(frac=args.sample)

thresholds_map1 = {'NK1': 8.5, 'PGP': 0.1, 'LOGD': 3, '3A4': 4.5, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 7, 'METAB': 40, 'OX1': 5.8, 'OX2': 6, 'PPB': 1, 'RAT_F': 1.0, 'TDI': 0, 'THROMBIN': 6}
thresholds_map2 = {'NK1': 9.5, 'PGP': 0.8, 'LOGD': 5, '3A4': 6, 'CB1': 8.0, 'DPP4': 7, 'HIVINT': 7, 'HIVPROT': 9, 'METAB': 70, 'OX1': 7.5, 'OX2': 8, 'PPB': 2, 'RAT_F': 1.9, 'TDI': 1, 'THROMBIN': 9}
threshold_1 = thresholds_map1[dataset_name]
threshold_2 = thresholds_map2[dataset_name]

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=15/100, shuffle=True) # split 15% as the test data, and the rest for train and calib (tc)

# fdp_nominals = np.round(np.linspace(0.1, 0.5, 9), 2) # nominal FDR levels
fdp_nominals = np.round(np.linspace(0.05, 0.95, 19), 2) # nominal FDR levels
all_res = pd.DataFrame() # results
all_res['fdp_nominals'] = fdp_nominals

epsilon = 1e-8
''' new two sheridan method '''
fdps_1new2, fdps_2new2, power1_new2, power2_new2 = [],[],[],[]
# the first step
# 15% test, 15% calibration2, 50% train, 20% calibration1
with Timer() as timer:
    mdl = get_model(args.model)
    mdl_rmse = get_model(args.model)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50 / 70, shuffle=True)
    mdl.fit(Xtrain, Ytrain)

    if args.model == 'rf':
        # for ensemble models such as rf, we use the predicted value and prediction variance as features for the error model
        # recommended by Sheridan et al. (2013)
        Ytrain_rmse_pred = mdl.predict(Xtrain_rmse)
        all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in mdl.estimators_])
        var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)

        mdl_rmse.fit(np.column_stack((Ytrain_rmse_pred, var_train_rmse)), np.abs(Ytrain_rmse - Ytrain_rmse_pred))
    else:
        # otherwise, above option is impossible and we simply use the original feature X as feature for the error model
        Ytrain_rmse_pred = mdl.predict(Xtrain_rmse)
        mdl_rmse.fit(Xtrain_rmse, np.abs(Ytrain_rmse - Ytrain_rmse_pred))

    # get z scores for calibration
    if args.model == 'rf':
        Ypred_calib = mdl.predict(Xcalib)
        all_Ypred = np.column_stack([tree.predict(Xcalib) for tree in mdl.estimators_])
        var_calib = np.var(all_Ypred, axis=1)
        rmse_calib = mdl_rmse.predict(np.column_stack((Ypred_calib, var_calib)))
    else:
        Ypred_calib = mdl.predict(Xcalib)
        rmse_calib = mdl_rmse.predict(Xcalib)

    z_calib1_1 = (threshold_1 - Ypred_calib) / (rmse_calib + epsilon)
    # z_calib1_2 = (np.maximum(threshold_1 - Ypred_calib1, Ypred_calib1 - threshold_2)) / rmse_calib1
    z_calib1_2 = (Ypred_calib - threshold_2) / (rmse_calib + epsilon)

    if args.model == 'rf':
        Ypred_test = mdl.predict(Xtest)
        all_Ypred = np.column_stack([tree.predict(Xtest) for tree in mdl.estimators_])
        var_test = np.var(all_Ypred, axis=1)
        rmse_test = mdl_rmse.predict(np.column_stack((Ypred_test, var_test)))
    else:
        Ypred_test = mdl.predict(Xtest)
        rmse_test = mdl_rmse.predict(Xtest)
    z_test_1 = (threshold_1 - Ypred_test) / (rmse_test + epsilon)
    z_test_2 = (Ypred_test - threshold_2) / (rmse_test + epsilon)
    min_R_1 = None  # Initialize as None
    min_R_2 = None

    for fdp_nominal in fdp_nominals:
        min_R_1 = None
        min_R_2 = None
        # --- Stage 1: Find min_R_1 ---
        for R in np.linspace(np.max(z_calib1_1), np.min(z_calib1_1), 200):
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
            try_fdr1, _, _, _ = eval_new(Ycalib, try_r_sel1, [], threshold_1, threshold_2)
            if np.any(fdp_nominal >= try_fdr1):
                # Since R is decreasing, the last value that satisfies the condition will be the minimum.
                min_R_1 = R

        # 2. Add fallback logic for min_R_1.
        # If the loop above failed to find a suitable R, use the most conservative threshold.
        if min_R_1 is None:
            min_R_1 = np.inf

        # --- Stage 2: Find min_R_2 ---
        # Since min_R_1 is now guaranteed to have a value, we can proceed to find min_R_2.
        for R in np.linspace(np.max(z_calib1_2), np.min(z_calib1_2), 300):
            # Note: try_r_sel1 here should use the already-determined min_R_1.
            try_r_sel1 = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= min_R_1]
            try_r_sel2 = [j for j in range(len(z_calib1_2)) if z_calib1_2[j] >= R]
            _, try_fdr2, _, _ = eval_new(Ycalib, try_r_sel1, try_r_sel2, threshold_1, threshold_2)
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

''' single stage '''
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
R_list = np.zeros(len(fdp_nominals))
with Timer() as timer:
    mdl = get_model(args.model)
    mdl_rmse = get_model(args.model)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50 / 70, shuffle=True)
    mdl.fit(Xtrain, Ytrain)

    if args.model == 'rf':
        # for ensemble models such as rf, we use the predicted value and prediction variance as features for the error model
        # recommended by Sheridan et al. (2013)
        Ytrain_rmse_pred = mdl.predict(Xtrain_rmse)
        all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in mdl.estimators_])
        var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)

        mdl_rmse.fit(np.column_stack((Ytrain_rmse_pred, var_train_rmse)), np.abs(Ytrain_rmse - Ytrain_rmse_pred))
    else:
        # otherwise, above option is impossible and we simply use the original feature X as feature for the error model
        Ytrain_rmse_pred = mdl.predict(Xtrain_rmse)
        mdl_rmse.fit(Xtrain_rmse, np.abs(Ytrain_rmse - Ytrain_rmse_pred))

    if args.model == 'rf':
        Ypred_calib = mdl.predict(Xcalib)
        all_Ypred = np.column_stack([tree.predict(Xcalib) for tree in mdl.estimators_])
        var_calib = np.var(all_Ypred, axis=1)
        rmse_calib = mdl_rmse.predict(np.column_stack((Ypred_calib, var_calib)))
    else:
        rmse_calib = mdl_rmse.predict(Xcalib)

    z_calib1_1 = (np.maximum(threshold_1 - Ypred_calib, Ypred_calib - threshold_2)) / (rmse_calib + epsilon)
    # z_calib1_2 = (Ypred_calib1 - threshold_2) / rmse_calib1

    for R in np.linspace(np.max(z_calib1_1), np.min(z_calib1_1), 200):
        try_r_sel = [j for j in range(len(z_calib1_1)) if z_calib1_1[j] >= R]
        try_fdr, _, _, _ = eval_n(Ycalib, try_r_sel, threshold_1, threshold_2)
        R_list[fdp_nominals >= try_fdr] = R

    # get z scores for test
    if args.model == 'rf':
        Ypred_test = mdl.predict(Xtest)
        all_Ypred = np.column_stack([tree.predict(Xtest) for tree in mdl.estimators_])
        var_test = np.var(all_Ypred, axis=1)
        rmse_test = mdl_rmse.predict(np.column_stack((Ypred_test, var_test)))
    else:
        rmse_test = mdl_rmse.predict(Xtest)

    z_test_1 = (np.maximum(threshold_1 - Ypred_test,Ypred_test - threshold_2 )) / (rmse_test + epsilon)
    # z_test_2 = (Ypred_test - threshold_2) / rmse_test

    # Search for the optimal R on the calibration set
    for i, R in enumerate(R_list):
        Sel_1 = [j for j in range(len(z_test_1)) if z_test_1[j] >= R]
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
    mdl = get_model(args.model)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50 / 85, shuffle=True)
    mdl.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000*((threshold_1 >= Ycalib) | (threshold_2 <= Ycalib)) - mdl.predict(Xcalib)  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -mdl.predict(Xtest)
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


# save the results
out_dir = os.path.join(f'result-model', args.model, f'{dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join(f'result-model', args.model, f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

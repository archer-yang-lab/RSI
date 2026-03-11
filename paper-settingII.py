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

fdp_nominals = np.linspace(0.1, 1.0, 9)
all_res = pd.DataFrame()
epsilon = 1e-8

all_res['fdp_nominals'] = fdp_nominals

''' single stage'''
def eval(Y, rejected, lower, higher):
    r"""
    Evaluate the selection correctness given the true values, the selected subsets, and the target region (lower,higher).

    Args:
        Y (np.ndarray): 1-d array of true responses (activity levels).
        rejected (np.ndarray): 1-d array of the indices of rejected hypotheses.
        lower, higher (float, float): The boundaries defining the desirable values of Y, which is (lower,higher).

    Returns:
        float, float, float: The empirical FDP, PCER and Power of the selection.
    """
    true_reject = np.sum((lower >= Y)|(Y >= higher))
    if len(rejected) == 0:
        fdp = 0
        pcer = 0
        power = 0
    else:
        fdp = np.sum((lower <= Y[rejected]) & (Y[rejected] < higher)) / len(rejected)
        pcer = np.sum((lower <= Y[rejected]) & (Y[rejected] < higher)) / len(Y)
        power = np.sum((lower > Y[rejected]) | (Y[rejected] >= higher)) / true_reject if true_reject != 0 else 0
    return fdp, pcer, power

''' single stage'''
def eval_n(Y, rejected_1, lower, higher):
    true_reject_1 = np.sum((lower >= Y)|(Y >= higher))
    if len(rejected_1) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum((lower < Y[rejected_1]) & (higher > Y[rejected_1])) / len(rejected_1)
        power = np.sum((lower >= Y[rejected_1])|(higher <= Y[rejected_1])) / true_reject_1 if true_reject_1 != 0 else 0
    return fdp, power

''' single stage conformal selection '''
fdpn_bcs, powern_bcs= [], []
fdpn_cs, powern_cs= [], []
from sklearn.ensemble import RandomForestClassifier
with Timer() as timer:
    rfc = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt',criterion="entropy")
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    #binary classification model
    rfc.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))
    #regression model
    rf.fit(Xtrain, Ytrain)
    Ycalib_pred = rf.predict(Xcalib)
    Ytest_pred = rf.predict(Xtest)
    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000*((threshold_1 >= Ycalib) | (threshold_2 <= Ycalib)) - rfc.predict_proba(Xcalib)[:, 1]  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -rfc.predict_proba(Xtest)[:, 1]
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_bcs.append(fdp)
        powern_bcs.append(power)

        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib) | (threshold_2 <= Ycalib)) - Ycalib_pred
        test_scores = -Ytest_pred
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs.append(fdp)
        powern_cs.append(power)


all_res['fdpn_bcs'] = fdpn_bcs
all_res['powern_bcs'] = powern_bcs

all_res['fdpn_cs'] = fdpn_cs
all_res['powern_cs'] = powern_cs

''' two stage conformal selection'''
fdpn_cs2union, powern_cs2union= [], []
with Timer() as timer:
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rf.fit(Xtrain, Ytrain)
    Ycalib_pred = rf.predict(Xcalib)
    Ytest_pred = rf.predict(Xtest)
    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000 * (threshold_1 >= Ycalib)  - Ycalib_pred
        test_scores = -Ytest_pred
        BH_2clipstep1 = BH(calib_scores_2clip, test_scores, fdp_nominal)

        calib_scores_2clip = 1000 * (threshold_2 <= Ycalib) - Ycalib_pred
        test_scores = -Ytest_pred
        BH_2clipstep2 = BH(calib_scores_2clip, test_scores, fdp_nominal)
        BH_2clip_union = np.intersect1d(BH_2clipstep1, BH_2clipstep2)
        BH_2clip = BH_2clip_union.astype(int)

        fdp, power = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs2union.append(fdp)
        powern_cs2union.append(power)

all_res['fdpn_cs2union'] = fdpn_cs2union
all_res['powern_cs2union'] = powern_cs2union

out_dir = os.path.join('result-union', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-union', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))


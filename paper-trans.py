from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
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

fdp_nominals = np.linspace(0.1, 1.0, 9)
all_res = pd.DataFrame()
all_res['fdp_nominals'] = fdp_nominals

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

fdpn_cssigned, powern_cssigned= [], []
fdpn_cs, powern_cs= [], []
fdpreg_cs, powerreg_cs= [], []
fdpbreg_cs, powerbreg_cs= [], []

with Timer() as timer:
    rfreg = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    rfreg.fit(Xtrain, Ytrain)
    Ypredreg_calib1 = rfreg.predict(Xcalib1)
    Ypredreg_test = rfreg.predict(Xtest)

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = Ycalib1 - Ypredreg_calib1
        test_scores = Ytest - Ypredreg_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power= eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cssigned.append(fdp)
        powern_cssigned.append(power)
all_res['time_cssigned'] = [timer.runtime] * len(fdp_nominals)

with Timer() as timer:
    rfcf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', criterion="entropy")
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    rfcf.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))
    Ypredcf_calib1 = rfcf.predict_proba(Xcalib1)[:, 1]
    Ypredcf_test = rfcf.predict_proba(Xtest)[:, 1]

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypredcf_calib1
        test_scores = -Ypredcf_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs.append(fdp)
        powern_cs.append(power)
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)

with Timer() as timer:
    rfbreg = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    rfbreg.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))
    Ypredbreg_calib1 = rfbreg.predict(Xcalib1)
    Ypredbreg_test = rfbreg.predict(Xtest)

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypredbreg_calib1
        test_scores = -Ypredbreg_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpbreg_cs.append(fdp)
        powerbreg_cs.append(power)
all_res['time_breg_cs'] = [timer.runtime] * len(fdp_nominals)

with Timer() as timer:
    rfreg = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    rfreg.fit(Xtrain, Ytrain)
    Ypredreg_calib1 = rfreg.predict(Xcalib1)
    Ypredreg_test = rfreg.predict(Xtest)

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) + Ypredreg_calib1
        test_scores = Ypredreg_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpreg_cs.append(fdp)
        powerreg_cs.append(power)
all_res['time_preg_cs'] = [timer.runtime] * len(fdp_nominals)

all_res['fdpn_cssigned'] = fdpn_cssigned
all_res['powern_cssigned'] = powern_cssigned

all_res['fdpn_cs'] = fdpn_cs
all_res['powern_cs'] = powern_cs

all_res['fdpreg_cs'] = fdpreg_cs
all_res['powerreg_cs'] = powerreg_cs

all_res['fdpbreg_cs'] = fdpbreg_cs
all_res['powerbreg_cs'] = powerbreg_cs

out_dir = os.path.join('result-trans', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-trans', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

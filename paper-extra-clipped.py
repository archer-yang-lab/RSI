from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import argparse
import time

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

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=15/100, shuffle=True) # tc: train and calib

# ofdp_nominals = np.linspace(0.1, 0.5, 9)
fdp_nominals = np.linspace(0.1, 1.0, 9)
all_res = pd.DataFrame()

# all_res['ofdp_nominals'] = ofdp_nominals
all_res['fdp_nominals'] = fdp_nominals

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

''' single stage conformal selection + signed error score'''
fdpn_cssigned, costn_cssigned, powern_cssigned, powers_cssigned= [], [], [], []
fdpn_cs, costn_cs, powern_cs, powers_cs= [], [], [], []
fdpreg_cs, costreg_cs, powerreg_cs, powersreg_cs= [], [], [], []
fdpbreg_cs, costbreg_cs, powerbreg_cs, powersbreg_cs= [], [], [], []
from sklearn.ensemble import RandomForestClassifier
with Timer() as timer:
    rfreg = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rfbreg = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rfcf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', criterion="entropy")
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    rfreg.fit(Xtrain, Ytrain)
    rfbreg.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))
    rfcf.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))

    Ypredreg_calib1 = rfreg.predict(Xcalib1)
    Ypredreg_test = rfreg.predict(Xtest)
    Ypredbreg_calib1 = rfbreg.predict(Xcalib1)
    Ypredbreg_test = rfbreg.predict(Xtest)
    Ypredcf_calib1 = rfcf.predict_proba(Xcalib1)[:, 1]
    Ypredcf_test = rfcf.predict_proba(Xtest)[:, 1]

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = Ycalib1 - Ypredreg_calib1
        test_scores = np.maximum(threshold_1-Ypredreg_test,Ypredreg_test-threshold_2)
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cssigned.append(fdp)
        powern_cssigned.append(power)
        costn_cssigned.append(cost)
        powers_cssigned.append(powers)

        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypredcf_calib1       # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -Ypredcf_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_cs.append(fdp)
        powern_cs.append(power)
        costn_cs.append(cost)
        powers_cs.append(powers)

        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypredbreg_calib1  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -Ypredbreg_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpbreg_cs.append(fdp)
        powerbreg_cs.append(power)
        costbreg_cs.append(cost)
        powersbreg_cs.append(powers)

        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) + Ypredreg_calib1  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = +Ypredreg_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpreg_cs.append(fdp)
        powerreg_cs.append(power)
        costreg_cs.append(cost)
        powersreg_cs.append(powers)

all_res['fdpn_cssigned'] = fdpn_cssigned
all_res['costn_cssigned'] = costn_cssigned
all_res['powern_cssigned'] = powern_cssigned
all_res['powers_cssigned'] = powers_cssigned
all_res['time_cssigned'] = [timer.runtime] * len(fdp_nominals)
all_res['fdpn_cs'] = fdpn_cs
all_res['costn_cs'] = costn_cs
all_res['powern_cs'] = powern_cs
all_res['powers_cs'] = powers_cs
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)
all_res['fdpreg_cs'] = fdpreg_cs
all_res['costreg_cs'] = costreg_cs
all_res['powerreg_cs'] = powerreg_cs
all_res['powersreg_cs'] = powersreg_cs
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)
all_res['fdpbreg_cs'] = fdpbreg_cs
all_res['costbreg_cs'] = costbreg_cs
all_res['powerbreg_cs'] = powerbreg_cs
all_res['powersbreg_cs'] = powersbreg_cs
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)

out_dir = os.path.join('result-signedc', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-signedc', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

''' single stage conformal selection + signed error score + uncertainty'''
fdpn_csunsigned, costn_csunsigned, powern_csunsigned, powers_csunsigned= [], [], [], []
fdpn_csun, costn_csun, powern_csun, powers_csun= [], [], [], []
fdpreg_csun, costreg_csun, powerreg_csun, powersreg_csun= [], [], [], []
epsilon = 1e-8
from sklearn.ensemble import RandomForestClassifier
with Timer() as timer:
    rfreg = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rfcf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', criterion="entropy")
    rf_rmse = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    Xtrain, Xcalib1, Ytrain, Ycalib1 = train_test_split(Xtc, Ytc, train_size=70 / 85, shuffle=True)
    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50 / 70, shuffle=True)

    rfreg.fit(Xtrain, Ytrain)
    Ytrain_rmse_predreg = rfreg.predict(Xtrain_rmse)
    all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in rfreg.estimators_])
    var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)
    rf_rmse.fit(np.column_stack((Ytrain_rmse_predreg, var_train_rmse)), np.abs(Ytrain_rmse - Ytrain_rmse_predreg))
    Ypredreg_calib1 = rfreg.predict(Xcalib1)
    all_Ypredreg_calib1 = np.column_stack([tree.predict(Xcalib1) for tree in rfreg.estimators_])
    var_calib1 = np.var(all_Ypredreg_calib1, axis=1)
    rmsereg_calib1 = rf_rmse.predict(np.column_stack((Ypredreg_calib1, var_calib1)))
    Ypredreg_test = rfreg.predict(Xtest)
    all_Ypredreg = np.column_stack([tree.predict(Xtest) for tree in rfreg.estimators_])
    var_test = np.var(all_Ypredreg, axis=1)
    rmsereg_test = rf_rmse.predict(np.column_stack((Ypredreg_test, var_test)))

    rfcf.fit(Xtrain, ((threshold_1 >= Ytrain) | (threshold_2 <= Ytrain)))
    Ytrain_rmse_predcf = rfcf.predict_proba(Xtrain_rmse)[:, 1]
    all_Ytrain_rmse_predcf = np.column_stack([tree.predict(Xtrain_rmse) for tree in rfcf.estimators_])
    var_train_rmsecf = np.var(all_Ytrain_rmse_predcf, axis=1)
    rf_rmse.fit(np.column_stack((Ytrain_rmse_predcf, var_train_rmsecf)),
                np.abs(((threshold_1 >= Ytrain_rmse) | (threshold_2 <= Ytrain_rmse)) - Ytrain_rmse_predcf))
    Ypredcf_calib1 = rfcf.predict_proba(Xcalib1)[:, 1]
    all_Ypredcf_calib1 = np.column_stack([tree.predict(Xcalib1) for tree in rfcf.estimators_])
    varcf_calib1 = np.var(all_Ypredcf_calib1, axis=1)
    rmserf_calib1 = rf_rmse.predict(np.column_stack((Ypredcf_calib1, varcf_calib1)))
    Ypredcf_test = rfcf.predict_proba(Xtest)[:, 1]
    all_Ypredcf = np.column_stack([tree.predict(Xtest) for tree in rfcf.estimators_])
    varcf_test = np.var(all_Ypredcf, axis=1)
    rmsecf_test = rf_rmse.predict(np.column_stack((Ypredcf_test, varcf_test)))

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores_2clip = (Ycalib1 - Ypredreg_calib1)/rmsereg_calib1
        test_scores = np.maximum(threshold_1 - Ypredreg_test, Ypredreg_test - threshold_2)/rmsereg_test
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_csunsigned.append(fdp)
        powern_csunsigned.append(power)
        costn_csunsigned.append(cost)
        powers_csunsigned.append(powers)

        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypredcf_calib1 / (rmserf_calib1  + epsilon)  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -Ypredcf_test / (rmsecf_test + epsilon)
        print(rmsecf_test[1:3])
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpn_csun.append(fdp)
        powern_csun.append(power)
        costn_csun.append(cost)
        powers_csun.append(powers)

        calib_scores_2clip = 1000 * ((threshold_1 >= Ycalib1) | (threshold_2 <= Ycalib1)) - Ypredreg_calib1 / (rmsereg_calib1 + epsilon)  # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
        test_scores = -Ypredreg_test / (rmsereg_test + epsilon)
        print(rmsereg_test[1:3])
        BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
        fdp, power, cost, powers = eval_n(Ytest, BH_2clip, threshold_1, threshold_2)
        fdpreg_csun.append(fdp)
        powerreg_csun.append(power)
        costreg_csun.append(cost)
        powersreg_csun.append(powers)

all_res['fdpn_csunsigned'] = fdpn_csunsigned
all_res['costn_csunsigned'] = costn_csunsigned
all_res['powern_csunsigned'] = powern_csunsigned
all_res['powers_csunsigned'] = powers_csunsigned
all_res['time_csunsigned'] = [timer.runtime] * len(fdp_nominals)
all_res['fdpn_csun'] = fdpn_csun
all_res['costn_csun'] = costn_csun
all_res['powern_csun'] = powern_csun
all_res['powers_csun'] = powers_csun
all_res['time_csun'] = [timer.runtime] * len(fdp_nominals)
all_res['fdpreg_csun'] = fdpreg_csun
all_res['costreg_csun'] = costreg_csun
all_res['powerreg_csun'] = powerreg_csun
all_res['powersreg_csun'] = powersreg_csun
all_res['time_csun'] = [timer.runtime] * len(fdp_nominals)

out_dir = os.path.join('result-signedc', f'{dataset_name} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result-signedc', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))

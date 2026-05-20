from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import argparse
import time
import os

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

# ----------------------------
# Timer
# ----------------------------
class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time


# ----------------------------
# Basic evaluation functions
# ----------------------------
def eval_inter(Y, selected, lower, higher):
    """
    Setting I:
    Selected set = predicted Indeterminate region.
    True positives = truly in (lower, higher).
    False discoveries = selected but actually in clear region.
    """
    selected = np.asarray(selected, dtype=int)
    true_indeterminate = np.sum((Y > lower) & (Y < higher))

    if len(selected) == 0:
        return 0.0, 0.0

    y_sel = Y[selected]
    fdp = np.mean((y_sel <= lower) | (y_sel >= higher))
    power = np.sum((y_sel > lower) & (y_sel < higher)) / true_indeterminate if true_indeterminate > 0 else 0.0
    return fdp, power


def eval_total_loss(Y, selected, lower, higher, c_pass=1.0, c_fail=10.0):
    """
    Total Loss for selected compounds in Setting I:
      - cost c_fail if selected but truly Y <= lower
      - cost c_pass if selected but truly Y >= higher
    """
    selected = np.asarray(selected, dtype=int)
    if len(selected) == 0:
        return 0.0

    y_sel = Y[selected]
    fail_cost = c_fail * np.sum(y_sel <= lower)
    pass_cost = c_pass * np.sum(y_sel >= higher)
    return float(fail_cost + pass_cost)


def selected_region_counts(Y, selected, lower, higher):
    """
    Count the true regions of selected compounds under Setting I.

    Returns
    -------
    n_selected_fail : int
        Number of selected compounds with Y <= lower.
    n_selected_ind : int
        Number of selected compounds with lower < Y < higher.
    n_selected_pass : int
        Number of selected compounds with Y >= higher.
    """
    selected = np.asarray(selected, dtype=int)
    if len(selected) == 0:
        return 0, 0, 0

    y_sel = Y[selected]
    n_selected_fail = int(np.sum(y_sel <= lower))
    n_selected_ind = int(np.sum((y_sel > lower) & (y_sel < higher)))
    n_selected_pass = int(np.sum(y_sel >= higher))
    return n_selected_fail, n_selected_ind, n_selected_pass


def BH(calib_scores, test_scores, q=0.1, rng=None):
    """
    Benjamini-Hochberg on conformal p-values.
    We keep the same '<' convention as in your current script.
    """
    if rng is None:
        rng = np.random.default_rng()

    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)

    for j in range(ntest):
        less_count = np.sum(calib_scores < test_scores[j])
        equal_count = np.sum(calib_scores == test_scores[j])
        pvals[j] = (less_count + rng.uniform() * (equal_count + 1)) / (ncalib + 1)

    df_test = pd.DataFrame({
        "id": np.arange(ntest),
        "score": test_scores,
        "pval": pvals
    }).sort_values(by="pval", kind="mergesort").reset_index(drop=True)

    df_test["threshold"] = q * np.arange(1, ntest + 1) / ntest
    eligible = np.where(df_test["pval"].to_numpy() <= df_test["threshold"].to_numpy())[0]

    if len(eligible) == 0:
        return np.array([], dtype=int)

    k = np.max(eligible)
    idx_sel = df_test.loc[:k, "id"].to_numpy(dtype=int)
    return idx_sel


def positive_class_proba(clf, X, positive_label=1):
    """
    Safe probability extraction in case a classifier sees only one class.
    """
    proba = clf.predict_proba(X)
    classes = clf.classes_
    if positive_label in classes:
        pos_idx = np.where(classes == positive_label)[0][0]
        return proba[:, pos_idx]
    return np.zeros(X.shape[0], dtype=float)


# ----------------------------
# Threshold maps
# ----------------------------
thresholds_map1 = {
    'NK1': 8.5, 'PGP': 0.1, 'LOGD': 3, '3A4': 4.5, 'CB1': 6.5,
    'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 7, 'METAB': 40, 'OX1': 5.8,
    'OX2': 6, 'PPB': 1, 'RAT_F': 1.0, 'TDI': 0, 'THROMBIN': 6
}
thresholds_map2 = {
    'NK1': 9.5, 'PGP': 0.8, 'LOGD': 5, '3A4': 6, 'CB1': 8.0,
    'DPP4': 7, 'HIVINT': 7, 'HIVPROT': 9, 'METAB': 70, 'OX1': 7.5,
    'OX2': 8, 'PPB': 2, 'RAT_F': 1.9, 'TDI': 1, 'THROMBIN': 9
}


# ----------------------------
# Helpers for EC threshold search
# ----------------------------
def choose_ec_threshold(z_calib, Ycalib, lower, higher, q):
    """
    Choose the smallest threshold R such that empirical FDP on calibration
    is <= q, under the rule select iff z >= R.
    """
    candidates = np.concatenate((
        np.array([np.inf]),
        np.sort(np.unique(z_calib))[::-1],
        np.array([-np.inf])
    ))

    best_R = np.inf
    for R in candidates:
        sel = np.where(z_calib >= R)[0]
        fdp, _ = eval_inter(Ycalib, sel, lower, higher)
        if fdp <= q:
            best_R = R
    return best_R


# ----------------------------
# Build artifacts once per trial
# ----------------------------
def build_trial_split(total_X, total_Y, seed):
    Xtc, Xtest, Ytc, Ytest = train_test_split(
        total_X, total_Y,
        test_size=15 / 100,
        shuffle=True,
        random_state=seed
    )
    return Xtc, Xtest, Ytc, Ytest


def build_ec_artifacts(Xtc, Xtest, Ytc, lower, higher, seed, n_estimators=100, max_depth=20, epsilon=1e-8):
    """
    Train the regression model, uncertainty model, and asymmetric-risk classifiers
    once for a trial, then reuse their predictions for all (q, lambda) combinations.
    """
    Xtrain_full, Xcalib1, Ytrain_full, Ycalib1 = train_test_split(
        Xtc, Ytc,
        train_size=70 / 85,
        shuffle=True,
        random_state=seed + 101
    )

    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(
        Xtrain_full, Ytrain_full,
        train_size=50 / 70,
        shuffle=True,
        random_state=seed + 102
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        random_state=seed + 103,
        n_jobs=N_JOBS
    )
    rf_rmse = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        random_state=seed + 104,
        n_jobs=N_JOBS
    )
    clf_fail = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        criterion='entropy',
        random_state=seed + 105,
        n_jobs=N_JOBS
    )
    clf_pass = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        criterion='entropy',
        random_state=seed + 106,
        n_jobs=N_JOBS
    )

    rf.fit(Xtrain, Ytrain)

    # Risk classifiers are trained only on the non-calibration part of the trial split.
    y_fail_train_full = (Ytrain_full <= lower).astype(int)
    y_pass_train_full = (Ytrain_full >= higher).astype(int)
    clf_fail.fit(Xtrain_full, y_fail_train_full)
    clf_pass.fit(Xtrain_full, y_pass_train_full)

    Ytrain_rmse_pred = rf.predict(Xtrain_rmse)
    all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in rf.estimators_])
    var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)

    rf_rmse.fit(
        np.column_stack((Ytrain_rmse_pred, var_train_rmse)),
        np.abs(Ytrain_rmse - Ytrain_rmse_pred)
    )

    Ypred_calib = rf.predict(Xcalib1)
    all_Ypred_calib = np.column_stack([tree.predict(Xcalib1) for tree in rf.estimators_])
    var_calib = np.var(all_Ypred_calib, axis=1)
    rmse_calib = rf_rmse.predict(np.column_stack((Ypred_calib, var_calib)))
    rmse_calib = np.maximum(rmse_calib, epsilon)

    Ypred_test = rf.predict(Xtest)
    all_Ypred_test = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
    var_test = np.var(all_Ypred_test, axis=1)
    rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
    rmse_test = np.maximum(rmse_test, epsilon)

    p_fail_calib = positive_class_proba(clf_fail, Xcalib1, positive_label=1)
    p_fail_test = positive_class_proba(clf_fail, Xtest, positive_label=1)
    p_pass_calib = positive_class_proba(clf_pass, Xcalib1, positive_label=1)
    p_pass_test = positive_class_proba(clf_pass, Xtest, positive_label=1)

    return {
        "Ycalib": Ycalib1,
        "Ypred_calib": Ypred_calib,
        "rmse_calib": rmse_calib,
        "Ypred_test": Ypred_test,
        "rmse_test": rmse_test,
        "p_fail_calib": p_fail_calib,
        "p_fail_test": p_fail_test,
        "p_pass_calib": p_pass_calib,
        "p_pass_test": p_pass_test
    }


def build_cs_artifacts(Xtc, Xtest, Ytc, lower, higher, seed, n_estimators=100, max_depth=20):
    """
    Train the main R1 classifier, fail-risk classifier, and pass-risk classifier
    once per trial, then reuse their probabilities for all (q, lambda) combinations.
    """
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(
        Xtc, Ytc,
        train_size=50 / 85,
        shuffle=True,
        random_state=seed + 201
    )

    y_inter_train = ((Ytrain > lower) & (Ytrain < higher)).astype(int)
    y_fail_train = (Ytrain <= lower).astype(int)
    y_pass_train = (Ytrain >= higher).astype(int)

    clf_inter = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        criterion='entropy',
        random_state=seed + 202,
        n_jobs=N_JOBS
    )
    clf_fail = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        criterion='entropy',
        random_state=seed + 203,
        n_jobs=N_JOBS
    )
    clf_pass = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        criterion='entropy',
        random_state=seed + 204,
        n_jobs=N_JOBS
    )

    clf_inter.fit(Xtrain, y_inter_train)
    clf_fail.fit(Xtrain, y_fail_train)
    clf_pass.fit(Xtrain, y_pass_train)

    pi_calib = positive_class_proba(clf_inter, Xcalib, positive_label=1)
    pi_test = positive_class_proba(clf_inter, Xtest, positive_label=1)

    p_fail_calib = positive_class_proba(clf_fail, Xcalib, positive_label=1)
    p_fail_test = positive_class_proba(clf_fail, Xtest, positive_label=1)
    p_pass_calib = positive_class_proba(clf_pass, Xcalib, positive_label=1)
    p_pass_test = positive_class_proba(clf_pass, Xtest, positive_label=1)

    ind_calib = ((Ycalib > lower) & (Ycalib < higher)).astype(float)

    return {
        "Ycalib": Ycalib,
        "pi_calib": pi_calib,
        "pi_test": pi_test,
        "p_fail_calib": p_fail_calib,
        "p_fail_test": p_fail_test,
        "p_pass_calib": p_pass_calib,
        "p_pass_test": p_pass_test,
        "ind_calib": ind_calib
    }




# ----------------------------
# Raw expected-cost multiplicative discount score helpers
# ----------------------------
SCORE_TYPE = "raw_expected_cost_multiplicative_discount"


def parse_float_grid(grid_string):
    """
    Parse eta grid for the raw expected-cost multiplicative discount.

    Accepted inputs
    ---------------
    1. A comma-separated numeric grid, e.g.
       '0,0.005,0.01,0.02,0.05,0.1'.
    2. One of the preset names:
       - 'focused':   fine grid for weak raw-cost discounting.
       - 'wide':      default grid for raw exposure C_tilde(x).
       - 'very_wide': stress-test grid for aggressive discounting.

    This version uses the unnormalized exposure
        C_tilde(x) = c_fail p_fail(x) + c_pass p_pass(x).
    With c_fail=10 and c_pass=1, eta must be much smaller than in the
    normalized version. For a pure fail-side risk, eta=0.10 gives weight
    exp(-1)=0.368; for a pure pass-side risk, eta=0.10 gives weight
    exp(-0.1)=0.905.
    """
    preset_grids = {
        "focused": "0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3",
        "wide": "0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5",
        "very_wide": "0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1",
    }

    grid_string = str(grid_string).strip()
    grid_key = grid_string.lower()
    if grid_key in preset_grids:
        grid_string = preset_grids[grid_key]

    values = []
    for item in grid_string.split(','):
        item = item.strip()
        if item:
            value = float(item)
            if value < 0:
                raise ValueError("eta values must be nonnegative for the multiplicative cost discount.")
            values.append(value)

    if len(values) == 0:
        raise ValueError("The grid string must contain at least one numeric value or a valid preset name.")

    # Keep the user-specified order but remove exact duplicates.
    unique_values = []
    seen = set()
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)

    return np.array(unique_values, dtype=float)


def raw_expected_cost_exposure(p_fail, p_pass, c_pass=1.0, c_fail=10.0):
    """
    Unnormalized predicted downstream cost exposure:

        C_tilde(x) = c_fail * p_fail(x) + c_pass * p_pass(x).

    This is the proposed score's raw cost scale. Fail-side risk receives weight
    c_fail, whereas pass-side risk receives weight c_pass. With c_fail > c_pass,
    fail-side candidates are discounted more strongly than pass-side candidates.

    The exposure is not divided by c_fail and is not upper-clipped. We only guard
    against small numerical negatives from probability estimates.
    """
    if c_fail <= 0 or c_pass < 0:
        raise ValueError("c_fail must be positive and c_pass must be nonnegative.")

    p_fail = np.asarray(p_fail, dtype=float)
    p_pass = np.asarray(p_pass, dtype=float)
    exposure = c_fail * p_fail + c_pass * p_pass
    return np.maximum(exposure, 0.0)


def cost_discount_factor(cost_exposure, eta, min_weight=1e-12):
    """
    Multiplicative risk discount based on raw cost exposure:

        omega_eta(x) = exp{-eta * C_tilde(x)},    0 < omega_eta(x) <= 1.

    Because C_tilde(x) is unnormalized, eta should usually be small. The lower
    bound prevents numerical overflow in the sign-aware EC score when z(x)<0 and
    omega_eta(x) is used in the denominator.
    """
    eta = float(eta)
    if eta < 0:
        raise ValueError("eta must be nonnegative for the multiplicative cost discount.")

    cost_exposure = np.asarray(cost_exposure, dtype=float)
    omega = np.exp(-eta * cost_exposure)
    return np.clip(omega, min_weight, 1.0)


def sign_aware_ec_score(base_z, omega):
    """
    Sign-aware multiplicative adjustment for the RSI-EC signed distance score:

        z_eta(x) = z(x) * omega_eta(x)          if z(x) > 0,
                 = 0                           if z(x) = 0,
                 = z(x) / omega_eta(x)         if z(x) < 0.

    Larger EC scores are better. This construction penalizes high-risk positive
    scores by shrinking them toward zero and penalizes high-risk negative scores
    by pushing them further below zero. It avoids the sign problem caused by
    blindly multiplying a negative z-score by a number in (0, 1].
    """
    base_z = np.asarray(base_z, dtype=float)
    omega = np.asarray(omega, dtype=float)

    z_eta = np.zeros_like(base_z, dtype=float)
    positive = base_z > 0
    negative = base_z < 0

    z_eta[positive] = base_z[positive] * omega[positive]
    z_eta[negative] = base_z[negative] / omega[negative]
    return z_eta


def discounted_indeterminate_probability(pi, omega):
    """
    Cost-discounted Indeterminate probability for RSI-CS:

        pi_eta(x) = pi(x) * omega_eta(x).

    Since pi(x) in [0, 1] and omega_eta(x) in (0, 1], the adjusted probability
    remains in [0, 1].
    """
    pi = np.asarray(pi, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return np.clip(pi * omega, 0.0, 1.0)


# ----------------------------
# Main simulation
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('sample', type=float)
    parser.add_argument('seed', type=int, help="base seed")
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--c_pass', type=float, default=1.0)
    parser.add_argument('--c_fail', type=float, default=10.0)
    parser.add_argument('--M', type=float, default=1000.0)
    parser.add_argument(
        '--eta_grid',
        type=str,
        default='wide',
        help=(
            "Eta grid for the raw expected-cost multiplicative discount. Use one of the presets "
            "focused, wide, very_wide, or provide a comma-separated numeric grid. "
            "Default: wide."
        )
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_name = args.dataset
    dataset_path = os.path.join('data', f'{dataset_name}_training_disguised.csv')
    dataset = pd.read_csv(dataset_path)

    if args.sample < 1:
        dataset = dataset.sample(frac=args.sample, random_state=args.seed)

    threshold_1 = thresholds_map1[dataset_name]
    threshold_2 = thresholds_map2[dataset_name]

    total_Y = dataset['Act'].to_numpy()
    total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

    q_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    eta_grid = parse_float_grid(args.eta_grid)

    # Keep the historical column names used by the plotting scripts:
    #   - EC tuning values are stored in gamma_ratio and lambda_ec.
    #   - CS tuning values are stored in lambda.
    gamma_ratio_grid = eta_grid
    lambda_grid = eta_grid

    trial_records = []

    with Timer() as total_timer:
        for trial in range(args.n_trials):
            trial_seed = args.seed + trial

            # Fair outer split shared by EC and CS in this trial
            Xtc, Xtest, Ytc, Ytest = build_trial_split(total_X, total_Y, seed=trial_seed)

            # Build reusable artifacts once per trial
            ec_art = build_ec_artifacts(Xtc, Xtest, Ytc, threshold_1, threshold_2, seed=trial_seed)
            cs_art = build_cs_artifacts(Xtc, Xtest, Ytc, threshold_1, threshold_2, seed=trial_seed)

            # ---------- RSI-EC: multiplicative cost-discount score ----------
            Ycalib_ec = ec_art["Ycalib"]
            Ypred_calib = ec_art["Ypred_calib"]
            rmse_calib = ec_art["rmse_calib"]
            Ypred_test = ec_art["Ypred_test"]
            rmse_test = ec_art["rmse_test"]
            p_fail_calib_ec = ec_art["p_fail_calib"]
            p_fail_test_ec = ec_art["p_fail_test"]
            p_pass_calib_ec = ec_art["p_pass_calib"]
            p_pass_test_ec = ec_art["p_pass_test"]

            base_z_calib = np.minimum(
                Ypred_calib - threshold_1,
                threshold_2 - Ypred_calib
            ) / rmse_calib
            base_z_test = np.minimum(
                Ypred_test - threshold_1,
                threshold_2 - Ypred_test
            ) / rmse_test

            # Raw predicted downstream cost exposure:
            #   C_tilde(x) = c_fail * p_fail(x) + c_pass * p_pass(x).
            # This version intentionally does not normalize by c_fail.
            cost_calib_ec = raw_expected_cost_exposure(
                p_fail_calib_ec, p_pass_calib_ec,
                c_pass=args.c_pass, c_fail=args.c_fail
            )
            cost_test_ec = raw_expected_cost_exposure(
                p_fail_test_ec, p_pass_test_ec,
                c_pass=args.c_pass, c_fail=args.c_fail
            )

            for eta_ec in gamma_ratio_grid:
                lambda_ec = eta_ec

                # Sign-aware multiplicative cost-discount EC score:
                #   omega_eta(x) = exp{-eta * C_tilde(x)},
                #   z_eta(x) = z(x) * omega_eta(x)       if z(x) > 0,
                #            = z(x) / omega_eta(x)       if z(x) < 0.
                # Select iff z_eta(x) >= R, with R calibrated by empirical FDP.
                omega_calib_ec = cost_discount_factor(cost_calib_ec, eta_ec)
                omega_test_ec = cost_discount_factor(cost_test_ec, eta_ec)
                z_calib = sign_aware_ec_score(base_z_calib, omega_calib_ec)
                z_test = sign_aware_ec_score(base_z_test, omega_test_ec)

                for q in q_grid:
                    R_hat = choose_ec_threshold(z_calib, Ycalib_ec, threshold_1, threshold_2, q)
                    selected = np.where(z_test >= R_hat)[0]

                    fdr, power = eval_inter(Ytest, selected, threshold_1, threshold_2)
                    n_selected = len(selected)
                    total_loss = eval_total_loss(
                        Ytest, selected, threshold_1, threshold_2,
                        c_pass=args.c_pass, c_fail=args.c_fail
                    )
                    average_cost = total_loss / n_selected if n_selected > 0 else 0.0
                    n_selected_fail, n_selected_ind, n_selected_pass = selected_region_counts(
                        Ytest, selected, threshold_1, threshold_2
                    )

                    trial_records.append({
                        "method": "RSI-EC",
                        "score_type": SCORE_TYPE,
                        "trial": trial,
                        "seed": trial_seed,
                        "q": q,
                        "eta": eta_ec,
                        "gamma_ratio": eta_ec,
                        "lambda_ec": lambda_ec,
                        "fdr": fdr,
                        "power": power,
                        "average_cost": average_cost,
                        "n_selected": n_selected,
                        "n_selected_fail": n_selected_fail,
                        "n_selected_ind": n_selected_ind,
                        "n_selected_pass": n_selected_pass
                    })

            # ---------- RSI-CS: multiplicative cost-discount score ----------
            Ycalib_cs = cs_art["Ycalib"]
            pi_calib = cs_art["pi_calib"]
            pi_test = cs_art["pi_test"]
            p_fail_calib = cs_art["p_fail_calib"]
            p_fail_test = cs_art["p_fail_test"]
            p_pass_calib = cs_art["p_pass_calib"]
            p_pass_test = cs_art["p_pass_test"]
            ind_calib = cs_art["ind_calib"]

            cost_calib_cs = raw_expected_cost_exposure(
                p_fail_calib, p_pass_calib,
                c_pass=args.c_pass, c_fail=args.c_fail
            )
            cost_test_cs = raw_expected_cost_exposure(
                p_fail_test, p_pass_test,
                c_pass=args.c_pass, c_fail=args.c_fail
            )

            for lam_idx, eta_cs in enumerate(lambda_grid):
                # Multiplicative cost-discount CS score:
                #   omega_eta(x) = exp{-eta * C_tilde(x)},
                #   pi_eta(x) = pi(x) * omega_eta(x),
                #   V_eta(x, y) = M * 1{y in R1} - pi_eta(x),
                #   Vhat_eta(x) = -pi_eta(x).
                M_eta = args.M
                omega_calib_cs = cost_discount_factor(cost_calib_cs, eta_cs)
                omega_test_cs = cost_discount_factor(cost_test_cs, eta_cs)
                pi_calib_eta = discounted_indeterminate_probability(pi_calib, omega_calib_cs)
                pi_test_eta = discounted_indeterminate_probability(pi_test, omega_test_cs)
                calib_scores = M_eta * ind_calib - pi_calib_eta
                test_scores = -pi_test_eta

                for q_idx, q in enumerate(q_grid):
                    bh_seed = args.seed + 100000 * trial + 1000 * lam_idx + 10 * q_idx
                    rng = np.random.default_rng(bh_seed)

                    selected = BH(calib_scores, test_scores, q=q, rng=rng)

                    fdr, power = eval_inter(Ytest, selected, threshold_1, threshold_2)
                    n_selected = len(selected)
                    total_loss = eval_total_loss(
                        Ytest, selected, threshold_1, threshold_2,
                        c_pass=args.c_pass, c_fail=args.c_fail
                    )
                    average_cost = total_loss / n_selected if n_selected > 0 else 0.0
                    n_selected_fail, n_selected_ind, n_selected_pass = selected_region_counts(
                        Ytest, selected, threshold_1, threshold_2
                    )

                    trial_records.append({
                        "method": "RSI-CS",
                        "score_type": SCORE_TYPE,
                        "trial": trial,
                        "seed": trial_seed,
                        "q": q,
                        "eta": eta_cs,
                        "gamma_ratio": np.nan,
                        "lambda_ec": np.nan,
                        "lambda": eta_cs,
                        "M_lambda": M_eta,
                        "fdr": fdr,
                        "power": power,
                        "average_cost": average_cost,
                        "n_selected": n_selected,
                        "n_selected_fail": n_selected_fail,
                        "n_selected_ind": n_selected_ind,
                        "n_selected_pass": n_selected_pass
                    })

    trial_df = pd.DataFrame(trial_records)

    # Summary table for plotting. The historical tuning columns are retained:
    #   - gamma_ratio is eta for RSI-EC.
    #   - lambda is eta for RSI-CS.
    summary_df = (
        trial_df
        .groupby(["method", "q", "gamma_ratio", "lambda"], dropna=False)
        .agg(
            mean_eta=("eta", "mean"),
            mean_fdr=("fdr", "mean"),
            std_fdr=("fdr", "std"),
            mean_power=("power", "mean"),
            std_power=("power", "std"),
            mean_average_cost=("average_cost", "mean"),
            std_average_cost=("average_cost", "std"),
            mean_n_selected=("n_selected", "mean"),
            std_n_selected=("n_selected", "std"),
            mean_n_selected_fail=("n_selected_fail", "mean"),
            std_n_selected_fail=("n_selected_fail", "std"),
            mean_n_selected_ind=("n_selected_ind", "mean"),
            std_n_selected_ind=("n_selected_ind", "std"),
            mean_n_selected_pass=("n_selected_pass", "mean"),
            std_n_selected_pass=("n_selected_pass", "std"),
            n_trials=("trial", "count")
        )
        .reset_index()
        .sort_values(["method", "q", "gamma_ratio", "lambda"])
    )

    out_dir = os.path.join(
        'result-cost-0518raw',
        f'{dataset_name} {args.sample:.2f}'
    )
    os.makedirs(out_dir, exist_ok=True)

    trial_df.to_csv(
        os.path.join(out_dir, f'{dataset_name} {args.sample:.2f} seed_{args.seed} trial_results.csv'),
        index=False
    )

    summary_df.to_csv(
        os.path.join(out_dir, f'{dataset_name} {args.sample:.2f} seed_{args.seed} summary_results.csv'),
        index=False
    )

    meta_df = pd.DataFrame({
        "dataset": [dataset_name],
        "sample": [args.sample],
        "base_seed": [args.seed],
        "n_trials": [args.n_trials],
        "c_pass": [args.c_pass],
        "c_fail": [args.c_fail],
        "M": [args.M],
        "eta_grid": [','.join(str(x) for x in eta_grid)],
        "score_type": [SCORE_TYPE],
        "runtime_seconds": [total_timer.runtime]
    })
    meta_df.to_csv(
        os.path.join(out_dir, f'{dataset_name} {args.sample:.2f} seed_{args.seed} meta.csv'),
        index=False
    )

    print(f"Finished. Results saved to: {out_dir}")
    print(f"Total runtime: {total_timer.runtime:.2f} seconds")


if __name__ == "__main__":
    main()

# RSI

**Beyond Go/No-Go Decisions: A Regional Selection Framework for Uncertainty-Aware Drug Screening**

## File Description

| File | Description |
|------|-------------|
| `paper-settingI.py` | Main experiment script for Setting I. |
| `paper-settingII.py` | Main experiment script for Setting II. |
| `paper-trans.py` | Main experiment script for different data transformations. |
| `paper-score.py` | Main experiment script for conformal selection score-function comparisons. |
| `paper-model.py` | Main experiment script for predictive model comparisons. |
| `paper-awarecost.py` | Main experiment script for the cost-aware RSI extension. It generates seed-level cost-aware results for both RSI-EC and RSI-CS. |
| `aggregate-cost-seeds.py` | Aggregates seed-level cost-aware `summary_results.csv` files into `aggregated_over_seeds.csv` for plotting. |
| `plot-settingI.py` | Plotting script for Setting I results. |
| `plot-settingII.py` | Plotting script for Setting II results. |
| `plot-trans.py` | Plotting script for transformation-related results. |
| `plot-score.py` | Plotting script for score-comparison results. |
| `plot-model.py` | Plotting script for model-comparison results. |
| `plot-cost.py` | Plotting script for cost-aware results across the 15 QSAR datasets. It can generate absolute cost/FDP/power plots and delta plots relative to `eta = 0`. |
| `plot-cost-three-panel.py` | Plotting script for representative cost-aware datasets. It generates row-wise three-panel figures showing cost/FDP/power, delta metrics, and selected-region composition changes. |

## Usage

### 1. Main experiments

The following scripts share the same command format:

- `paper-settingI.py`
- `paper-settingII.py`
- `paper-trans.py`
- `paper-score.py`

Run an experiment with:

```bash
python <script_name>.py CB1 1.0 42
```

Generate the corresponding plot with:

```bash
python <plot_script>.py 100
```

#### Examples

```bash
python paper-settingI.py CB1 1.0 42
python plot-settingI.py 100

python paper-settingII.py CB1 1.0 42
python plot-settingII.py 100

python paper-trans.py CB1 1.0 42
python plot-trans.py 100

python paper-score.py CB1 1.0 42
python plot-score.py 100
```

#### Argument Description

For experiment scripts:

- `<dataset_name>`: name of the dataset, for example `CB1`
- `<sample_ratio>`: sampling ratio, for example `1.0`
- `<seed>`: random seed for reproducibility, for example `42`

For plotting scripts:

- `<n_itr>`: number of iterations, for example `100`

### 2. Predictive model comparison

Run the model comparison experiment with:

```bash
python paper-model.py CB1 1.0 42 nn
```

Generate the corresponding plot with:

```bash
python plot-model.py 100
```

#### Argument Description

For `paper-model.py`:

- `<dataset_name>`: name of the dataset, for example `CB1`
- `<sample_ratio>`: sampling ratio, for example `1.0`
- `<seed>`: random seed for reproducibility, for example `42`
- `<model>`: predictive model, for example `nn` (`nn`, `lin`, `rf`)

For `plot-model.py`:

- `<n_itr>`: number of iterations, for example `100`

### 3. Cost-aware RSI extension

The cost-aware extension studies Setting I when false discoveries from the two clear regions have asymmetric downstream costs. In the current implementation, a selected compound with true response `Y <= c1` receives cost `c_fail`, while a selected compound with true response `Y >= c2` receives cost `c_pass`. The default setting is `c_fail = 10` and `c_pass = 1`.

The main experiment script is:

- `paper-awarecost.py`

This script runs both RSI-EC and RSI-CS with a cost-aware score. The current score type is `raw_expected_cost_multiplicative_discount`. It uses the raw predicted cost exposure

```text
C_tilde(x) = c_fail * p_fail(x) + c_pass * p_pass(x)
```

and the multiplicative discount

```text
omega_eta(x) = exp{-eta * C_tilde(x)}.
```

For RSI-EC, the discount is applied through a sign-aware adjustment of the EC distance score. For RSI-CS, the discount is applied to the estimated Indeterminate probability score. The tuning parameter `eta` controls the strength of the cost discount. Because `C_tilde(x)` is not normalized, the useful `eta` values are usually small.

Run one cost-aware experiment with:

```bash
python paper-awarecost.py <dataset_name> <sample_ratio> <seed> --eta_grid wide
```

For example:

```bash
python paper-awarecost.py CB1 1.00 42 --eta_grid wide
```

The preset `wide` grid is:

```text
0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5
```

The script saves results to:

```text
result-cost-0518raw/<dataset_name> <sample_ratio>/
```

For each seed, it produces:

```text
<dataset_name> <sample_ratio> seed_<seed> trial_results.csv
<dataset_name> <sample_ratio> seed_<seed> summary_results.csv
<dataset_name> <sample_ratio> seed_<seed> meta.csv
```

The most important output columns are:

- `method`: `RSI-EC` or `RSI-CS`
- `q`: nominal FDP level
- `eta`: cost-scaling parameter
- `gamma_ratio`: historical EC tuning column, equal to `eta` for RSI-EC
- `lambda`: historical CS tuning column, equal to `eta` for RSI-CS
- `mean_fdr`: mean FDP over trials within one seed-level summary
- `mean_power`: mean power over trials within one seed-level summary
- `mean_average_cost`: mean selected-set average cost
- `mean_n_selected_fail`: mean number of selected compounds from the true Fail region
- `mean_n_selected_ind`: mean number of selected compounds from the true Indeterminate region
- `mean_n_selected_pass`: mean number of selected compounds from the true Pass region

After running all seeds, aggregate the seed-level summaries with:

```bash
python aggregate-cost-seeds.py \
  --base_dir result-cost-0518raw \
  --sample 1.00 \
  --expected_n_files 100
```

This creates one file per dataset:

```text
result-cost-0518raw/<dataset_name> 1.00/<dataset_name> 1.00 aggregated_over_seeds.csv
```

This aggregated file is the input used by the cost-aware plotting scripts.

#### Plot all 15 datasets

Use `plot-cost.py` to generate method-separated figures over the 15 QSAR datasets. This script can draw the absolute cost/FDP/power plots, the delta plots relative to `eta = 0`, or both.

```bash
python plot-cost.py 1.00 \
  --result_dir result-cost-0518raw \
  --output_dir figure-cost-0518raw \
  --q_values 0.2 0.3 \
  --methods RSI-EC RSI-CS \
  --eta_grid 0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5 \
  --plot_type both
```

The output files are saved under:

```text
figure-cost-0518raw/sample_1.00/
```

Typical output filenames include:

```text
combined_ec_split_average_cost_q02.pdf
combined_cs_split_average_cost_q02.pdf
combined_ec_delta_cost_power_q02.pdf
combined_cs_delta_cost_power_q02.pdf
```

#### Plot representative three-panel figures

Use `plot-cost-three-panel.py` to generate row-wise representative figures. Each dataset occupies one row, and the three columns show:

1. average cost, FDP, and power as `eta` varies;
2. cost reduction, change in power, and change in FDP relative to `eta = 0`;
3. changes in selected-region counts relative to `eta = 0`, namely Fail, Indeterminate, and Pass selections.

For the representative datasets used in the cost-aware analysis, run:

```bash
python plot-cost-three-panel.py 1.00 \
  --result_dir result-cost-0518raw \
  --output_dir figure-cost-0518raw-three-panel \
  --datasets PPB CB1 DPP4 OX2 \
  --q_values 0.2 0.3 \
  --methods RSI-EC RSI-CS \
  --eta_grid 0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5
```

The output files are saved under:

```text
figure-cost-0518raw-three-panel/sample_1.00/
```

#### Notes on consistency

For the current raw-cost implementation, always keep the `eta_grid` used in the plotting scripts consistent with the grid used in `paper-awarecost.py`. The plotting scripts filter rows by the supplied `eta_grid`; if the plotting grid does not match the experiment grid, some available results may be omitted from the figures.

For the current raw-cost results, also pass `--base_dir result-cost-0518raw` when running `aggregate-cost-seeds.py`, because its internal default directory may refer to an older experiment folder.


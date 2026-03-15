# RSI
Beyond Go/No-Go Decisions: A  Regional Selection Framework for Uncertainty-Aware Drug Screening

## File Description

| File | Description |
|------|-------------|
| `paper-model.py` | Main experiment script for predictive model comparisons. |
| `paper-score.py` | Main experiment script for conformal selection score-function comparisons. |
| `paper-settingI.py` | Main experiment script for Setting I. |
| `paper-settingII.py` | Main experiment script for Setting II. |
| `paper-trans.py` | Main experiment script for different data transformations. |
| `plot-model.py` | Plotting script for model-comparison results. |
| `plot-score.py` | Plotting script for score-comparison results. |
| `plot-settingI.py` | Plotting script for Setting I results. |
| `plot-settingII.py` | Plotting script for Setting II results. |
| `plot-trans.py` | Plotting script for transformation-related results. |

## Usage

Run the script with:

```bash
###for trans, score,settingI,settingII
python paper-trans.py CB1 1.0 42
python plot-trans.py 100
### Argument Description
- `<dataset_name>`: name of the dataset, for example `CB1`
- `<sample_ratio>`: sampling ratio, for example `1.0`
- `<seed>`: random seed for reproducibility, for example `42`
- `<n_itr>`: number of iterations, for example `100`
###for model
python paper-model.py CB1 1.0 42 nn
python plot-model.py 100
### Argument Description
- `<dataset_name>`: name of the dataset, for example `CB1`
- `<sample_ratio>`: sampling ratio, for example `1.0`
- `<seed>`: random seed for reproducibility, for example `42`
- `<model>`: model for prediction, for example `nn` (nn, lin, rf)
- `<n_itr>`: number of iterations, for example `100`

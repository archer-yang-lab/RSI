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
| `plot-settingI.py` | Plotting script for Setting I results. |
| `plot-settingII.py` | Plotting script for Setting II results. |
| `plot-trans.py` | Plotting script for transformation-related results. |
| `plot-score.py` | Plotting script for score-comparison results. |
| `plot-model.py` | Plotting script for model-comparison results. |

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

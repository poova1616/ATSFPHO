# ATSFPHO — Advanced Time Series Forecasting with Prophet + Optuna

**Contents**
- data/ - contains synthetic_timeseries.csv
- src/  - source modules (dataset generator, cv, optimizer, trainer, baseline, viz)
- outputs/ - generated outputs (forecasts, best_params)
- main.py - orchestrates the pipeline

## Quickstart

1. Create and activate a Python virtual environment.
2. Install requirements: `pip install -r requirements.txt`
3. Run the pipeline: `python main.py`

Notes:
- Optuna optimization will perform multiple Prophet fits; reduce `n_trials` in `main.py` if you want faster runs.
- Prophet installation may require a C++ build tool on Windows or a wheel. See https://facebook.github.io/prophet/docs/installation.html

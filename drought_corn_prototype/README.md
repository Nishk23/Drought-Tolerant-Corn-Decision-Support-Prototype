# Drought-Tolerant Corn Decision Support Prototype

## Overview

This repository provides a compact prototype for decision support focused on evaluating and ranking corn varieties under drought conditions. It demonstrates a reproducible data pipeline, exploratory analysis, a basic modeling workflow, and report generation to help users explore variety responses to drought scenarios.

The materials are intended for demonstration and experimentation rather than validated, production-ready deployment.

## What's Included

- Synthetic example trial data and a clear example schema
- Jupyter notebook detailing the workflow and visual analyses
- Scripts to run the pipeline and produce tabular and visual outputs
- A minimal web app to browse generated outputs interactively

## Repository Structure

- `drought_corn_prototype/`
	- `app.py` — Minimal app to preview outputs interactively.
	- `build_prototype.py` — Runs data prep, model training, and report generation.
	- `requirements.txt` — Python package dependencies.
	- `data/synthetic_corn_trials.csv` — Example dataset used for demonstrations.
	- `notebooks/01_drought_corn_prototype.ipynb` — Exploratory notebook with code and narrative.
	- `outputs/` — Generated results: figures and summary tables.

## Quickstart

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r drought_corn_prototype/requirements.txt
```

2. Run the pipeline to generate outputs:

```bash
python drought_corn_prototype/build_prototype.py
```

3. Explore the step-by-step analysis in the notebook:

```bash
jupyter lab drought_corn_prototype/notebooks/01_drought_corn_prototype.ipynb
```

4. (Optional) Launch the lightweight app to view results interactively:

```bash
FLASK_APP=drought_corn_prototype/app.py flask run
# open http://127.0.0.1:5000 in your browser
```

Notes:

- The example dataset is synthetic and provided for demonstration. Replace `data/synthetic_corn_trials.csv` with real trial data to run with your own inputs.
- `build_prototype.py` writes outputs to `drought_corn_prototype/outputs/`.

## Data Requirements

The prototype expects a tidy CSV containing trial observations by variety, with columns for identifiers (variety, site, year), environmental descriptors, and response variables such as yield or drought-score proxies. The notebook shows example column names and preprocessing steps.

When using your own data, ensure consistent column names or update the preprocessing code in `build_prototype.py`.

## Outputs

After running the pipeline, check `drought_corn_prototype/outputs/` for:

- `tables/variety_ranking.csv` — ranked varieties by selected performance metric
- `tables/feature_importance.csv` — feature importance or driver summaries
- `tables/model_metrics.csv` — model evaluation numbers (e.g., RMSE, R2)
- `figures/` — visualizations generated during analysis

## Extending the Prototype

Ideas for next steps:

- Add data validation and configuration to support multiple input schemas
- Introduce more advanced modeling pipelines and cross-validation
- Add automated tests and a CI workflow to ensure reproducibility
- Replace the minimal app with a more feature-rich dashboard if needed

Contributions are welcome via pull requests; please include a description of changes and any test updates.

## License

This repository is provided as an illustrative prototype. Add an appropriate open-source license if you plan to publish or distribute.

## Contact

Open an issue in this repository for questions, bug reports, or enhancement requests.

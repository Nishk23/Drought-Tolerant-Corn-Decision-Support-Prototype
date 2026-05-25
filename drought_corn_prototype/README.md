# Drought-Tolerant Corn Decision Support Prototype

This project is a lightweight, presentation-ready proof of concept for a 10-minute KWS Assessment Center talk on how to structure a data science workflow for drought-tolerant corn.

It uses synthetic representative data only. It does not rely on real KWS proprietary trial, breeding, genomic, or weather data, and it should not be treated as a validated production model.

## What is included

- Synthetic trial data with multiple varieties, locations, years, drought stress levels, soil types, weather variables, remote-sensing indicators, and trait proxies.
- An end-to-end Jupyter Notebook that creates the data, labels drought stress, runs exploratory analysis, trains a Random Forest regressor, evaluates performance, explains the drivers, and builds a breeder-ready ranking.
- Clean PNG figures suitable for PowerPoint slides.
- CSV outputs for the synthetic dataset, model metrics, and breeder ranking.
- A simple Streamlit dashboard for interactive exploration.

## Why synthetic data

The goal is to demonstrate project logic and decision support structure without claiming access to real field, breeding, genomic, or weather data. The synthetic dataset is designed to be realistic enough for communication and workflow demonstration, not scientific validation.

## Project workflow

1. Generate synthetic trial records across varieties, locations, and years.
2. Apply rule-based drought stress labels: Normal, Moderate Stress, and Severe Stress.
3. Engineer drought-related features from weather, soil, remote sensing, and trait proxies.
4. Train a Random Forest regressor to predict drought tolerance score.
5. Validate on the most recent year and on a simple unseen-location split.
6. Produce feature importance and breeder-ready recommendations.

## How to run the notebook

From the repository root:

```bash
pip install -r drought_corn_prototype/requirements.txt
jupyter notebook drought_corn_prototype/notebooks/01_drought_corn_prototype.ipynb
```

Run all cells to regenerate:

- `data/synthetic_corn_trials.csv`
- `outputs/tables/model_metrics.csv`
- `outputs/tables/variety_ranking.csv`
- `outputs/tables/feature_importance.csv`
- `outputs/figures/*.png`

## How to run the Streamlit app

After the notebook has generated the outputs:

```bash
streamlit run drought_corn_prototype/app.py
```

The app reads the saved synthetic data and presentation assets.

## Slide mapping suggestion

- Slide 1: Project purpose and disclaimer.
- Slide 2: Synthetic data design and drought labeling logic.
- Slide 3: Exploratory visuals showing drought distribution, rainfall vs soil moisture, and yield by stress level.
- Slide 4: Model validation on unseen year and unseen location.
- Slide 5: Feature importance and key drought drivers.
- Slide 6: Breeder-ready variety ranking and suggested actions.
- Slide 7: Closing note on limitations and next steps.

## Presentation note

This prototype is intentionally conservative. It shows how a decision-support pipeline could be structured, but it does not claim production readiness, biological certainty, or direct transferability to KWS proprietary data.
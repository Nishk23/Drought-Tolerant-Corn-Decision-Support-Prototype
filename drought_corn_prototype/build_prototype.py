from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
STRESS_ORDER = ['Normal', 'Moderate Stress', 'Severe Stress']
# Updated colors: Normal=green, Moderate=orange, Severe=red
STRESS_COLORS = {'Normal': '#2ca02c', 'Moderate Stress': '#ff7f0e', 'Severe Stress': '#d62728'}


def safe_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def project_paths():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / 'data'
    fig_dir = project_dir / 'outputs' / 'figures'
    table_dir = project_dir / 'outputs' / 'tables'
    for directory in [data_dir, fig_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return project_dir, data_dir, fig_dir, table_dir


def build_reference_tables(seed=RANDOM_SEED):
    local_rng = np.random.default_rng(seed)
    location_rows = [
        {'location': 'North Field 1', 'region': 'North', 'soil_type': 'Loam', 'base_rainfall_mm': 790, 'base_temp_c': 26.4, 'base_soil_moisture_pct': 28, 'water_holding_capacity': 29.0, 'soil_organic_matter': 3.2, 'location_drought_offset': -0.05, 'yield_bonus': 0.35},
        {'location': 'North Field 2', 'region': 'North', 'soil_type': 'Clay Loam', 'base_rainfall_mm': 760, 'base_temp_c': 26.8, 'base_soil_moisture_pct': 27, 'water_holding_capacity': 30.0, 'soil_organic_matter': 3.4, 'location_drought_offset': -0.03, 'yield_bonus': 0.25},
        {'location': 'North Field 3', 'region': 'North', 'soil_type': 'Silt Loam', 'base_rainfall_mm': 730, 'base_temp_c': 27.0, 'base_soil_moisture_pct': 26, 'water_holding_capacity': 28.0, 'soil_organic_matter': 3.0, 'location_drought_offset': 0.00, 'yield_bonus': 0.18},
        {'location': 'Central Field 1', 'region': 'Central', 'soil_type': 'Loam', 'base_rainfall_mm': 650, 'base_temp_c': 28.0, 'base_soil_moisture_pct': 24, 'water_holding_capacity': 25.0, 'soil_organic_matter': 2.8, 'location_drought_offset': 0.03, 'yield_bonus': 0.08},
        {'location': 'Central Field 2', 'region': 'Central', 'soil_type': 'Sandy Loam', 'base_rainfall_mm': 610, 'base_temp_c': 28.5, 'base_soil_moisture_pct': 22, 'water_holding_capacity': 22.0, 'soil_organic_matter': 2.2, 'location_drought_offset': 0.05, 'yield_bonus': -0.02},
        {'location': 'Central Field 3', 'region': 'Central', 'soil_type': 'Clay Loam', 'base_rainfall_mm': 670, 'base_temp_c': 27.8, 'base_soil_moisture_pct': 25, 'water_holding_capacity': 27.0, 'soil_organic_matter': 2.9, 'location_drought_offset': 0.02, 'yield_bonus': 0.12},
        {'location': 'East Field 1', 'region': 'East', 'soil_type': 'Sandy Loam', 'base_rainfall_mm': 530, 'base_temp_c': 29.4, 'base_soil_moisture_pct': 19, 'water_holding_capacity': 18.0, 'soil_organic_matter': 1.9, 'location_drought_offset': 0.09, 'yield_bonus': -0.12},
        {'location': 'East Field 2', 'region': 'East', 'soil_type': 'Loam', 'base_rainfall_mm': 560, 'base_temp_c': 29.0, 'base_soil_moisture_pct': 20, 'water_holding_capacity': 20.0, 'soil_organic_matter': 2.2, 'location_drought_offset': 0.08, 'yield_bonus': -0.08},
        {'location': 'East Field 3', 'region': 'East', 'soil_type': 'Clay Loam', 'base_rainfall_mm': 545, 'base_temp_c': 29.2, 'base_soil_moisture_pct': 21, 'water_holding_capacity': 21.0, 'soil_organic_matter': 2.1, 'location_drought_offset': 0.07, 'yield_bonus': -0.06},
        {'location': 'West Field 1', 'region': 'West', 'soil_type': 'Sandy Clay', 'base_rainfall_mm': 470, 'base_temp_c': 30.0, 'base_soil_moisture_pct': 17, 'water_holding_capacity': 16.0, 'soil_organic_matter': 1.7, 'location_drought_offset': 0.10, 'yield_bonus': -0.20},
        {'location': 'West Field 2', 'region': 'West', 'soil_type': 'Sandy Loam', 'base_rainfall_mm': 500, 'base_temp_c': 29.8, 'base_soil_moisture_pct': 18, 'water_holding_capacity': 17.0, 'soil_organic_matter': 1.8, 'location_drought_offset': 0.10, 'yield_bonus': -0.16},
        {'location': 'West Field 3', 'region': 'West', 'soil_type': 'Loam', 'base_rainfall_mm': 530, 'base_temp_c': 29.6, 'base_soil_moisture_pct': 19, 'water_holding_capacity': 18.0, 'soil_organic_matter': 2.0, 'location_drought_offset': 0.08, 'yield_bonus': -0.10},
    ]
    location_frame = pd.DataFrame(location_rows)

    variety_rows = []
    for index in range(1, 31):
        genetic = float(np.clip(local_rng.normal(72, 8), 50, 92))
        root = float(np.clip(genetic + local_rng.normal(0, 6), 50, 95))
        stay_green = float(np.clip(genetic + local_rng.normal(0, 7), 48, 94))
        maturity_group = local_rng.choice(['Early', 'Medium', 'Late'], p=[0.3, 0.45, 0.25])
        variety_rows.append({
            'variety_id': f'V{index:03d}',
            'genetic_drought_score': round(genetic, 1),
            'root_depth_score': round(root, 1),
            'stay_green_score': round(stay_green, 1),
            'maturity_group': maturity_group,
        })
    variety_frame = pd.DataFrame(variety_rows)
    return location_frame, variety_frame


def assign_drought_label(row):
    severe = (
        row['drought_pressure_score'] >= 46
        or (
            row['soil_moisture_pct'] < 16
            and row['rainfall_deficit'] > 120
            and row['heat_stress_days'] >= 6
            and row['yield_t_ha'] < 8.5
        )
    )
    if severe:
        return 'Severe Stress'
    moderate = (
        row['drought_pressure_score'] >= 30
        or row['soil_moisture_pct'] < 23
        or row['rainfall_deficit'] > 70
        or row['heat_stress_days'] >= 4
        or row['canopy_temperature_c'] > 32.0
    )
    if moderate:
        return 'Moderate Stress'
    return 'Normal'


def engineer_features(frame, seed=RANDOM_SEED):
    engineered = frame.copy()
    engineered['moisture_stress_index'] = np.clip(
        0.6 * (1 - engineered['soil_moisture_pct'] / 45.0)
        + 0.4 * (engineered['rainfall_deficit'] / (engineered['rainfall_mm'] + engineered['rainfall_deficit'] + 1)),
        0,
        1,
    )
    engineered['heat_risk_index'] = np.clip(
        0.5 * (engineered['heat_stress_days'] / 18.0)
        + 0.5 * np.clip((engineered['temperature_avg_c'] - 27.0) / 11.0, 0, 1),
        0,
        1,
    )
    engineered['vegetation_health_index'] = np.clip(
        0.65 * engineered['ndvi']
        + 0.35 * np.clip(1 - (engineered['canopy_temperature_c'] - 24.0) / 22.0, 0, 1),
        0,
        1,
    )
    engineered['trait_resilience_score'] = np.clip(
        0.4 * engineered['genetic_drought_score']
        + 0.35 * engineered['root_depth_score']
        + 0.25 * engineered['stay_green_score'],
        0,
        100,
    )
    engineered['drought_pressure_score'] = np.clip(
        100 * (
            0.45 * engineered['moisture_stress_index']
            + 0.25 * engineered['heat_risk_index']
            + 0.15 * (1 - engineered['ndvi'])
            + 0.15 * np.clip((engineered['canopy_temperature_c'] - 28.0) / 14.0, 0, 1)
        ),
        0,
        100,
    )
    engineered['yield_stability_score'] = np.clip(
        100 * (
            0.55 * (engineered['trait_resilience_score'] / 100.0)
            + 0.25 * engineered['vegetation_health_index']
            + 0.20 * (1 - engineered['moisture_stress_index'])
        ),
        0,
        100,
    )
    yield_component = np.interp(engineered['yield_t_ha'], [1.0, 14.5], [20, 95])
    noise = np.random.default_rng(seed).normal(0, 2.2, len(engineered))
    engineered['drought_tolerance_score'] = np.clip(
        0.32 * yield_component
        + 0.22 * engineered['trait_resilience_score']
        + 0.10 * engineered['soil_moisture_pct']
        + 18 * engineered['vegetation_health_index']
        + 0.08 * engineered['water_holding_capacity']
        + 0.08 * engineered['soil_organic_matter'] * 10
        + 0.04 * engineered['rainfall_mm'] / 10
        - 1.5 * engineered['heat_stress_days']
        - 0.9 * (engineered['canopy_temperature_c'] - 28)
        - 0.05 * engineered['rainfall_deficit']
        + engineered['location_yield_bonus'] * 4
        + noise,
        0,
        100,
    )
    engineered['drought_stress_label'] = engineered.apply(assign_drought_label, axis=1)
    engineered['recommended_action'] = np.select(
        [
            engineered['drought_tolerance_score'] >= 78,
            engineered['drought_tolerance_score'] >= 65,
            engineered['drought_tolerance_score'] >= 50,
        ],
        [
            np.where(engineered['drought_stress_label'] == 'Severe Stress', 'Retest under dry-site trials', 'Advance'),
            'Retest under dry-site trials',
            'Monitor',
        ],
        default='Do not prioritize',
    )
    return engineered


def regression_summary(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mse)),
        'r2': r2_score(y_true, y_pred),
    }


def aggregate_feature_importance(model_pipeline):
    transformed_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model_pipeline.named_steps['model'].feature_importances_
    rows = []
    for name, importance in zip(transformed_names, importances):
        if name.startswith('num__'):
            base = name.replace('num__', '')
        elif name.startswith('cat__'):
            remainder = name.replace('cat__', '')
            if '_' in remainder:
                base = remainder.split('_')[0]
            else:
                base = remainder
        else:
            base = name
        rows.append({'base_feature': base, 'importance': importance})
    return pd.DataFrame(rows).groupby('base_feature', as_index=False)['importance'].sum().sort_values('importance', ascending=False)


def save_drought_distribution(frame, fig_dir):
    counts = frame['drought_stress_label'].value_counts().reindex(STRESS_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=[STRESS_COLORS[label] for label in counts.index])
    ax.set_title('Drought stress label distribution')
    ax.set_ylabel('Trial records')
    ax.set_xlabel('Drought stress label')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for index, value in enumerate(counts.values):
        ax.text(index, value + 10, f'{int(value)}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / 'drought_label_distribution.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_rainfall_vs_soil_moisture(frame, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label in STRESS_ORDER:
        subset = frame[frame['drought_stress_label'] == label]
        ax.scatter(
            subset['rainfall_mm'],
            subset['soil_moisture_pct'],
            s=20,
            alpha=0.7,
            label=label,
            color=STRESS_COLORS[label],
            edgecolors='none',
        )
    ax.set_title('Prototype: Drought Labels from Rainfall & Soil Moisture')
    ax.set_xlabel('Rainfall (mm)')
    ax.set_ylabel('Soil moisture (%)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(fig_dir / 'rainfall_vs_soil_moisture.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_yield_by_stress_level(frame, fig_dir):
    data = [frame.loc[frame['drought_stress_label'] == label, 'yield_t_ha'].values for label in STRESS_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    box = ax.boxplot(data, patch_artist=True, labels=STRESS_ORDER, medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, label in zip(box['boxes'], STRESS_ORDER):
        patch.set_facecolor(STRESS_COLORS[label])
        patch.set_alpha(0.7)
    ax.set_title('Yield by drought stress level')
    ax.set_xlabel('Drought stress label')
    ax.set_ylabel('Yield (t/ha)')
    fig.tight_layout()
    fig.savefig(fig_dir / 'yield_by_stress_level.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_feature_importance_plot(aggregated_importance, fig_dir):
    top = aggregated_importance.head(12).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(top['base_feature'], top['importance'], color='#2563eb')
    ax.set_title('Random Forest feature importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(fig_dir / 'feature_importance.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_variety_ranking_plot(ranking_frame, fig_dir):
    top = ranking_frame.head(10).copy()
    display_frame = top[[
        'variety_id',
        'predicted_drought_tolerance_score',
        'average_yield_under_stress',
        'yield_stability_score',
        'main_strength',
        'suggested_action',
    ]].copy()
    display_frame['predicted_drought_tolerance_score'] = display_frame['predicted_drought_tolerance_score'].round(1)
    display_frame['average_yield_under_stress'] = display_frame['average_yield_under_stress'].round(1)
    display_frame['yield_stability_score'] = display_frame['yield_stability_score'].round(1)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    table = ax.table(cellText=display_frame.values, colLabels=display_frame.columns, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    ax.set_title('Top drought-tolerant varieties and suggested actions', pad=20)
    fig.tight_layout()
    fig.savefig(fig_dir / 'variety_ranking.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def generate_dataset(data_dir):
    location_frame, variety_frame = build_reference_tables()
    year_profile = {2020: 0.18, 2021: 0.24, 2022: 0.58, 2023: 0.33, 2024: 0.50}
    year_temp_trend = {2020: 0.0, 2021: 0.2, 2022: 0.7, 2023: 0.4, 2024: 0.9}
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for year in [2020, 2021, 2022, 2023, 2024]:
        for _, location in location_frame.iterrows():
            for _, variety in variety_frame.iterrows():
                for replicate in [1, 2]:
                    drought_severity = np.clip(year_profile[year] + location['location_drought_offset'] + rng.normal(0, 0.06), 0, 1)
                    rainfall_mm = max(45, location['base_rainfall_mm'] * (1 - 0.55 * drought_severity) + rng.normal(0, 35))
                    rainfall_deficit = max(0, location['base_rainfall_mm'] + 70 * year_profile[year] - rainfall_mm)
                    temperature_avg_c = location['base_temp_c'] + year_temp_trend[year] + 4.0 * drought_severity + rng.normal(0, 0.7)
                    heat_stress_days = int(np.clip(rng.poisson(lam=max(0.6, 1.5 + 5.5 * drought_severity + 0.12 * (temperature_avg_c - 28))), 0, 18))
                    soil_moisture_pct = float(np.clip(location['base_soil_moisture_pct'] - 12 * drought_severity + 0.015 * (rainfall_mm - location['base_rainfall_mm']) + rng.normal(0, 2.2), 6, 42))
                    water_holding_capacity = float(np.clip(location['water_holding_capacity'] + rng.normal(0, 0.7), 8, 40))
                    soil_organic_matter = float(np.clip(location['soil_organic_matter'] + rng.normal(0, 0.14), 0.8, 6.5))
                    ndvi = float(np.clip(0.88 - 0.28 * drought_severity + 0.004 * soil_moisture_pct + 0.0015 * (variety['stay_green_score'] - 70) + 0.0008 * (variety['root_depth_score'] - 70) + rng.normal(0, 0.025), 0.12, 0.95))
                    canopy_temperature_c = float(np.clip(29 + 0.18 * (temperature_avg_c - 28) + 4.5 * drought_severity - 0.025 * variety['root_depth_score'] - 0.02 * soil_moisture_pct + rng.normal(0, 0.9), 24, 46))
                    biomass_index = float(np.clip(36 + 62 * ndvi + 0.28 * soil_moisture_pct + 0.03 * variety['genetic_drought_score'] - 1.1 * heat_stress_days + rng.normal(0, 3.5), 12, 120))
                    maturity_shift = {'Early': -1.5, 'Medium': 0.0, 'Late': 2.5}[variety['maturity_group']]
                    flowering_days = float(np.clip(73 + 0.11 * heat_stress_days + 2.8 * drought_severity + maturity_shift + rng.normal(0, 1.4), 68, 110))
                    plant_height_cm = float(np.clip(102 + 1.55 * biomass_index + 0.18 * variety['root_depth_score'] - 0.5 * heat_stress_days + rng.normal(0, 8), 85, 260))
                    yield_t_ha = float(np.clip(
                        2.2
                        + 0.007 * rainfall_mm
                        + 0.11 * soil_moisture_pct
                        + 1.9 * ndvi
                        - 0.13 * canopy_temperature_c
                        - 0.11 * heat_stress_days
                        + 0.018 * variety['genetic_drought_score']
                        + 0.016 * variety['root_depth_score']
                        + 0.014 * variety['stay_green_score']
                        + location['yield_bonus'] * 4
                        + 0.25 * (year - 2020)
                        + rng.normal(0, 0.65),
                        1.0,
                        14.5,
                    ))
                    records.append({
                        'trial_id': f'TRIAL_{len(records) + 1:05d}',
                        'variety_id': variety['variety_id'],
                        'location': location['location'],
                        'region': location['region'],
                        'year': year,
                        'soil_type': location['soil_type'],
                        'rainfall_mm': round(rainfall_mm, 1),
                        'rainfall_deficit': round(rainfall_deficit, 1),
                        'temperature_avg_c': round(temperature_avg_c, 1),
                        'heat_stress_days': heat_stress_days,
                        'soil_moisture_pct': round(soil_moisture_pct, 1),
                        'water_holding_capacity': round(water_holding_capacity, 1),
                        'soil_organic_matter': round(soil_organic_matter, 2),
                        'ndvi': round(ndvi, 3),
                        'canopy_temperature_c': round(canopy_temperature_c, 1),
                        'biomass_index': round(biomass_index, 1),
                        'flowering_days': round(flowering_days, 1),
                        'plant_height_cm': round(plant_height_cm, 1),
                        'maturity_group': variety['maturity_group'],
                        'genetic_drought_score': round(variety['genetic_drought_score'], 1),
                        'root_depth_score': round(variety['root_depth_score'], 1),
                        'stay_green_score': round(variety['stay_green_score'], 1),
                        'yield_t_ha': round(yield_t_ha, 2),
                        'location_yield_bonus': location['yield_bonus'],
                        'replicate': replicate,
                    })

    raw_trials = pd.DataFrame(records)
    trial_frame = engineer_features(raw_trials)
    trial_frame = trial_frame.drop(columns=['location_yield_bonus', 'replicate'])
    ordered_columns = [
        'trial_id', 'variety_id', 'location', 'region', 'year', 'soil_type',
        'rainfall_mm', 'rainfall_deficit', 'temperature_avg_c', 'heat_stress_days',
        'soil_moisture_pct', 'water_holding_capacity', 'soil_organic_matter', 'ndvi',
        'canopy_temperature_c', 'biomass_index', 'flowering_days', 'plant_height_cm',
        'maturity_group', 'genetic_drought_score', 'root_depth_score', 'stay_green_score',
        'yield_t_ha', 'drought_stress_label', 'drought_tolerance_score', 'recommended_action',
        'moisture_stress_index', 'heat_risk_index', 'vegetation_health_index',
        'yield_stability_score', 'drought_pressure_score', 'trait_resilience_score',
    ]
    trial_frame = trial_frame[ordered_columns]
    trial_frame.to_csv(data_dir / 'synthetic_corn_trials.csv', index=False)
    return trial_frame


def train_and_score(frame, fig_dir, table_dir):
    feature_frame = frame.copy()
    most_recent_year = feature_frame['year'].max()
    train_frame = feature_frame[feature_frame['year'] < most_recent_year].copy()
    test_frame = feature_frame[feature_frame['year'] == most_recent_year].copy()
    unseen_location = feature_frame['location'].value_counts().index[-1]
    location_train_frame = feature_frame[(feature_frame['year'] < most_recent_year) & (feature_frame['location'] != unseen_location)].copy()
    location_test_frame = feature_frame[(feature_frame['year'] < most_recent_year) & (feature_frame['location'] == unseen_location)].copy()

    numeric_features = [
        'rainfall_mm', 'rainfall_deficit', 'temperature_avg_c', 'heat_stress_days',
        'soil_moisture_pct', 'water_holding_capacity', 'soil_organic_matter',
        'ndvi', 'canopy_temperature_c', 'biomass_index', 'flowering_days', 'plant_height_cm',
        'genetic_drought_score', 'root_depth_score', 'stay_green_score',
        'moisture_stress_index', 'heat_risk_index', 'vegetation_health_index',
        'trait_resilience_score', 'year',
    ]
    categorical_features = ['region', 'soil_type', 'maturity_group']
    model_features = numeric_features + categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', safe_one_hot_encoder())]), categorical_features),
        ],
    )
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, min_samples_leaf=2, n_jobs=-1)),
    ])

    model_pipeline.fit(train_frame[model_features], train_frame['drought_tolerance_score'])
    test_frame['predicted_drought_tolerance_score'] = model_pipeline.predict(test_frame[model_features])

    location_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, min_samples_leaf=2, n_jobs=-1)),
    ])
    location_model.fit(location_train_frame[model_features], location_train_frame['drought_tolerance_score'])
    location_test_frame['predicted_drought_tolerance_score'] = location_model.predict(location_test_frame[model_features])

    metrics_rows = []
    metrics_rows.append({'split': 'unseen_year', 'group': 'overall', 'n_samples': len(test_frame), **regression_summary(test_frame['drought_tolerance_score'], test_frame['predicted_drought_tolerance_score'])})
    for label in STRESS_ORDER:
        subset = test_frame[test_frame['drought_stress_label'] == label]
        if not subset.empty:
            metrics_rows.append({'split': 'unseen_year', 'group': label, 'n_samples': len(subset), **regression_summary(subset['drought_tolerance_score'], subset['predicted_drought_tolerance_score'])})
    metrics_rows.append({'split': 'unseen_location', 'group': unseen_location, 'n_samples': len(location_test_frame), **regression_summary(location_test_frame['drought_tolerance_score'], location_test_frame['predicted_drought_tolerance_score'])})
    for label in STRESS_ORDER:
        subset = location_test_frame[location_test_frame['drought_stress_label'] == label]
        if not subset.empty:
            metrics_rows.append({'split': 'unseen_location', 'group': f'{unseen_location} | {label}', 'n_samples': len(subset), **regression_summary(subset['drought_tolerance_score'], subset['predicted_drought_tolerance_score'])})
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(table_dir / 'model_metrics.csv', index=False)

    aggregated_importance = aggregate_feature_importance(model_pipeline)
    aggregated_importance.to_csv(table_dir / 'feature_importance.csv', index=False)
    save_feature_importance_plot(aggregated_importance, fig_dir)

    latest_scores = test_frame.groupby('variety_id', as_index=False).agg(
        predicted_drought_tolerance_score=('predicted_drought_tolerance_score', 'mean'),
        yield_stability_score=('yield_stability_score', 'mean'),
    )
    stress_yield = feature_frame[feature_frame['drought_stress_label'].isin(['Moderate Stress', 'Severe Stress'])].groupby('variety_id', as_index=False)['yield_t_ha'].mean().rename(columns={'yield_t_ha': 'average_yield_under_stress'})
    variety_summary = feature_frame[['variety_id', 'genetic_drought_score', 'root_depth_score', 'stay_green_score', 'maturity_group']].drop_duplicates().merge(latest_scores, on='variety_id', how='left').merge(stress_yield, on='variety_id', how='left')
    variety_summary['average_yield_under_stress'] = variety_summary['average_yield_under_stress'].fillna(feature_frame.groupby('variety_id')['yield_t_ha'].mean().mean())

    def main_strength(row):
        trait_values = {
            'Deep-rooted adaptation': row['root_depth_score'],
            'Stay-green canopy retention': row['stay_green_score'],
            'Genetic drought tolerance': row['genetic_drought_score'],
        }
        return max(trait_values, key=trait_values.get)

    def recommended_environment(row):
        if row['root_depth_score'] >= row['stay_green_score'] and row['root_depth_score'] >= row['genetic_drought_score']:
            return 'Dry-site and low-rainfall trials'
        if row['stay_green_score'] >= row['root_depth_score'] and row['stay_green_score'] >= row['genetic_drought_score']:
            return 'Hot, late-season stress environments'
        return 'Broad rainfed environments'

    advance_cutoff = variety_summary['predicted_drought_tolerance_score'].quantile(0.75)
    retest_cutoff = variety_summary['predicted_drought_tolerance_score'].quantile(0.50)
    monitor_cutoff = variety_summary['predicted_drought_tolerance_score'].quantile(0.30)
    stability_cutoff = variety_summary['yield_stability_score'].quantile(0.50)

    def suggested_action(row):
        if row['predicted_drought_tolerance_score'] >= advance_cutoff and row['yield_stability_score'] >= stability_cutoff:
            return 'Advance'
        if row['predicted_drought_tolerance_score'] >= retest_cutoff:
            return 'Retest under dry-site trials'
        if row['predicted_drought_tolerance_score'] >= monitor_cutoff:
            return 'Monitor'
        return 'Do not prioritize'

    variety_summary['main_strength'] = variety_summary.apply(main_strength, axis=1)
    variety_summary['recommended_environment'] = variety_summary.apply(recommended_environment, axis=1)
    variety_summary['suggested_action'] = variety_summary.apply(suggested_action, axis=1)
    variety_summary = variety_summary.sort_values('predicted_drought_tolerance_score', ascending=False)
    variety_summary = variety_summary[[
        'variety_id', 'predicted_drought_tolerance_score', 'average_yield_under_stress', 'yield_stability_score',
        'main_strength', 'recommended_environment', 'suggested_action',
    ]].copy()
    variety_summary['predicted_drought_tolerance_score'] = variety_summary['predicted_drought_tolerance_score'].round(1)
    variety_summary['average_yield_under_stress'] = variety_summary['average_yield_under_stress'].round(1)
    variety_summary['yield_stability_score'] = variety_summary['yield_stability_score'].round(1)
    variety_summary.to_csv(table_dir / 'variety_ranking.csv', index=False)
    save_variety_ranking_plot(variety_summary, fig_dir)

    return {
        'metrics_frame': metrics_frame,
        'variety_summary': variety_summary,
        'test_frame': test_frame,
        'location_test_frame': location_test_frame,
    }


def generate_plots(frame, fig_dir):
    save_drought_distribution(frame, fig_dir)
    save_rainfall_vs_soil_moisture(frame, fig_dir)
    save_yield_by_stress_level(frame, fig_dir)


def main():
    _, data_dir, fig_dir, table_dir = project_paths()
    trials = generate_dataset(data_dir)
    generate_plots(trials, fig_dir)
    results = train_and_score(trials, fig_dir, table_dir)
    print(f'Synthetic dataset created: {trials.shape[0]:,} rows x {trials.shape[1]} columns')
    print(trials['drought_stress_label'].value_counts().reindex(STRESS_ORDER))
    print('\nUnseen year metrics:')
    print(results['metrics_frame'][results['metrics_frame']['split'] == 'unseen_year'])
    print('\nUnseen location held out:')
    print(results['metrics_frame'][results['metrics_frame']['split'] == 'unseen_location'])
    print('\nTop breeding candidates:')
    print(results['variety_summary'].head(10))
    return results


if __name__ == '__main__':
    main()

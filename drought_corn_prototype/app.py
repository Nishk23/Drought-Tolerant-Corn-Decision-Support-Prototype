from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / 'data' / 'synthetic_corn_trials.csv'
RANKING_PATH = PROJECT_DIR / 'outputs' / 'tables' / 'variety_ranking.csv'
IMPORTANCE_PATH = PROJECT_DIR / 'outputs' / 'tables' / 'feature_importance.csv'


st.set_page_config(
    page_title='Drought-Tolerant Corn Decision Support Prototype',
    layout='wide',
)

st.title('Drought-Tolerant Corn Decision Support Prototype')
st.caption('Synthetic representative data for project demonstration only')


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_ranking() -> pd.DataFrame:
    return pd.read_csv(RANKING_PATH)


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCE_PATH)


def format_metric(value, digits=1):
    return f'{value:,.{digits}f}'


if not DATA_PATH.exists() or not RANKING_PATH.exists() or not IMPORTANCE_PATH.exists():
    st.warning('Run the notebook first to generate the synthetic data and output tables.')
    st.stop()


df = load_data()
ranking = load_ranking()
importance = load_importance().sort_values('importance', ascending=False)

years = sorted(df['year'].unique().tolist())
locations = sorted(df['location'].unique().tolist())
stress_labels = ['Normal', 'Moderate Stress', 'Severe Stress']

st.sidebar.header('Filters')
selected_years = st.sidebar.multiselect('Year', years, default=years)
selected_locations = st.sidebar.multiselect('Location', locations, default=locations)
selected_labels = st.sidebar.multiselect('Drought stress label', stress_labels, default=stress_labels)

filtered = df[
    df['year'].isin(selected_years)
    & df['location'].isin(selected_locations)
    & df['drought_stress_label'].isin(selected_labels)
].copy()

if filtered.empty:
    st.info('No records match the current filters. Expand the selection to continue.')
    st.stop()


available_varieties = filtered['variety_id'].unique().tolist()
ranking_subset = ranking[ranking['variety_id'].isin(available_varieties)].copy()
top_predicted_variety = ranking_subset.sort_values('predicted_drought_tolerance_score', ascending=False).iloc[0]['variety_id']


metric_cols = st.columns(4)
metric_cols[0].metric('Trial records', f'{len(filtered):,}')
metric_cols[1].metric('Average yield (t/ha)', format_metric(filtered['yield_t_ha'].mean()))
metric_cols[2].metric('Severe drought records', f"{int((filtered['drought_stress_label'] == 'Severe Stress').sum()):,}")
metric_cols[3].metric('Top predicted variety', top_predicted_variety)


st.subheader('Drought label distribution')
label_counts = filtered['drought_stress_label'].value_counts().reindex(stress_labels).fillna(0).reset_index()
label_counts.columns = ['drought_stress_label', 'count']
fig_label = px.bar(
    label_counts,
    x='drought_stress_label',
    y='count',
    color='drought_stress_label',
    color_discrete_map={'Normal': '#2ca02c', 'Moderate Stress': '#ff7f0e', 'Severe Stress': '#d62728'},
    title='Distribution of drought stress labels',
)
fig_label.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_label, use_container_width=True)


st.subheader('Rainfall vs soil moisture')
fig_scatter = px.scatter(
    filtered,
    x='rainfall_mm',
    y='soil_moisture_pct',
    color='drought_stress_label',
    color_discrete_map={'Normal': '#2ca02c', 'Moderate Stress': '#ff7f0e', 'Severe Stress': '#d62728'},
    title='Prototype: Drought Labels from Rainfall & Soil Moisture',
    hover_data=['variety_id', 'location', 'year', 'yield_t_ha'],
    opacity=0.72,
)
fig_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_scatter, use_container_width=True)


st.subheader('Feature importance')
top_importance = importance.head(12)
fig_importance = px.bar(
    top_importance.sort_values('importance', ascending=True),
    x='importance',
    y='feature',
    orientation='h',
    title='Key drivers of drought tolerance from the trained model',
    color='importance',
    color_continuous_scale=['#dbeafe', '#2563eb'],
)
fig_importance.update_layout(coloraxis_showscale=False, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_importance, use_container_width=True)


st.subheader('Top recommended varieties')
st.dataframe(
    ranking_subset.head(10)[
        [
            'variety_id',
            'predicted_drought_tolerance_score',
            'average_yield_under_stress',
            'yield_stability_score',
            'main_strength',
            'recommended_environment',
            'suggested_action',
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
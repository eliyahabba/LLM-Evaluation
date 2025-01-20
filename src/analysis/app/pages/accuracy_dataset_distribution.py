import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")


# ---------------------------------------------------------------------
# Utility functions to load dimension-based data
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Loaders for model, dataset, overall data
# ---------------------------------------------------------------------
def load_shots_models_datasets(base_dir: Path):
    """Load available shots, models and their datasets from the given base_dir."""
    hierarchy = {}  # shots -> model -> datasets
    if not os.path.isdir(base_dir):
        return hierarchy

    for shots_dir in os.listdir(base_dir):
        if not shots_dir.startswith("Shots_"):
            continue
        shots_path = os.path.join(base_dir, shots_dir)
        if not os.path.isdir(shots_path):
            continue

        # Initialize shots level
        shots_key = shots_dir.split("_")[-1]
        hierarchy[shots_key] = {}

        for model in os.listdir(shots_path):
            model_path = os.path.join(shots_path, model)
            if not os.path.isdir(model_path):
                continue

            # Initialize model level
            hierarchy[shots_key][model] = set()

            # Add datasets for this model
            for dataset in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset)
                if os.path.isdir(dataset_path):
                    hierarchy[shots_key][model].add(dataset)

    return hierarchy


def load_data(base_dir, shots, model):
    """Load parquet (quad-level accuracy data) and the 'quad' distance matrix for a model/dataset."""
    model_dir = os.path.join(base_dir, f"Shots_{shots}", model)
    parquet_path = os.path.join(model_dir, 'model_statistics.parquet')
    config_data = pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else None
    return config_data


# ---------------------------------------------------------------------
# Streamlit main
# ---------------------------------------------------------------------

def generate_model_performance_comparison1(
        dataset_statistics: pd.DataFrame,
        model_name: str,
        shots_selected: int,
) -> None:
    """
    Generate performance comparison plots with configurations count.

    Args:
        df: DataFrame containing performance data with columns ['model', 'dataset', 'score'].
        model_name: Name of the model being analyzed.
        shots_selected: Number of shots used in the experiment.
        base_results_dir: Directory to save the plots.
    """
    print(f"Starting performance comparison for model: {model_name}...")

    # Save statistics to parquet

    # Generate bar plot with error bars and configuration counts
    fig = px.bar(
        dataset_statistics,
        x='dataset',
        y='average_accuracy',
        error_y='std_dev_accuracy',
        title=f'Model Performance: {model_name} (Shots: {shots_selected})',
        labels={
            'dataset': 'Dataset',
            'average_accuracy': 'Average Accuracy',
            'std_dev_accuracy': 'Standard Deviation'
        },
        text='configuration_count'  # Show number of configurations on bars
    )

    fig.update_layout(
        xaxis_tickangle=45,
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
        height=600,
        width=1000
    )

    # Format the configuration count text
    fig.update_traces(
        texttemplate='n=%{text}',  # Add 'n=' prefix to configuration count
        textposition='outside',  # Position text above bars
        textfont=dict(size=12)  # Adjust text size
    )
    return fig


def generate_model_performance_comparison(
        dataset_statistics: pd.DataFrame,
        model_name: str,
        shots_selected: int
) -> None:
    """
    Generate boxplots showing performance distribution across different configurations for each dataset.
    """
    print(f"Starting performance comparison for model: {model_name}...")

    # Reset index if it's not already in the right format
    if isinstance(dataset_statistics.index, pd.MultiIndex):
        dataset_statistics = dataset_statistics.reset_index()

    # Create a violin plot or box plot showing distribution
    fig = px.box(
        dataset_statistics,
        x='dataset',
        y='accuracy',
        title=f'Model Performance Distribution: {model_name} (Shots: {shots_selected})',
        labels={
            'dataset': 'Dataset',
            'accuracy': 'Accuracy',
        },
        hover_data=['template', 'separator', 'enumerator', 'choices_order', 'count']
        # Show configuration details on hover
    )

    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        width=1200,
        height=600,
        title_x=0.5,
        margin=dict(l=50, r=50, t=50, b=100),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            tickangle=45,  # Rotate dataset names
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            range=[0, 1]  # Set range for accuracy
        ),
        boxmode='overlay'
    )

    # Update boxplot appearance
    fig.update_traces(
        fillcolor='lightblue',
        line=dict(color='navy'),
        boxpoints='all',  # Show all points
        jitter=0.3,  # Add some random spread to the points
        pointpos=-1.8,  # Position points slightly to the left
        marker=dict(
            size=4,
            opacity=0.6
        )
    )

    # Add total configuration count for each dataset
    # configs_per_dataset = dataset_statistics.groupby('dataset')['count'].sum()
    # for dataset in fig.data[0].x:
    #     fig.add_annotation(
    #         x=dataset,
    #         y=1.02,  # Position above the plot
    #         text=f"n={int(configs_per_dataset[dataset])}",
    #         showarrow=False,
    #         font=dict(size=10)
    #     )

    return fig


# Add this at the beginning of your app.py, right after the imports:
def add_description_style():
    st.markdown("""
        <style>
        .graph-description {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 5px solid #4a90e2;
        }
        .graph-description-title {
            color: #1f77b4;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


def load_data_all_datasets(base_dir, shots, model):
    """Load parquet (quad-level accuracy data) and the 'quad' distance matrix for a model/dataset."""
    model_dir = os.path.join(base_dir, f"Shots_{shots}", model)
    parquet_path = os.path.join(model_dir, 'model_statistics_all_datasets.parquet')
    config_data = pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else None
    return config_data


def generate_model_performance_comparison_all_datasets(config_data_all_models, shots_selected):
    min_num_of_configurations = st.radio(
        "Minimum sample threshold per configuration (configurations must meet this threshold when counting samples across all datasets combined)",
        [100, 500, 1000, 1500, 2000, 2500, 3000], index=1)
    all_dataset_statistics = []
    for model, config_data in config_data_all_models.items():
        # Filter by minimum configurations
        config_data = config_data[config_data['count'] >= min_num_of_configurations]

        # Calculate statistics
        stats = {
            'model': model,
            'average_accuracy': round(config_data["accuracy"].mean(), 4),
            'std': round(config_data["accuracy"].std(), 4),
            'configuration_count': int(config_data["accuracy"].count())
        }
        all_dataset_statistics.append(stats)

    # Create a single DataFrame
    stats_df = pd.DataFrame(all_dataset_statistics)

    # Optional: Sort by accuracy if desired
    stats_df = stats_df.sort_values('average_accuracy', ascending=False)

    # Create improved figure
    fig = px.bar(
        stats_df,
        x='average_accuracy',
        y='model',
        error_x='std',
        orientation='h',  # Horizontal bars
        title=f'Model Performance Comparison (Shots: {shots_selected})',
        labels={
            'model': '',  # Remove y-axis label
            'average_accuracy': 'Average Accuracy',
            'std': 'Standard Deviation'
        },
        text='configuration_count'
    )

    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor='white',
        width=1000,
        height=400,
        title_x=0.5,  # Center title
        margin=dict(l=150, r=50, t=50, b=50),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            range=[0, max(stats_df['average_accuracy']) * 1.15]  # Add some padding
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        )
    )

    # Update bar appearance
    fig.update_traces(
        marker_color='lightblue',
        marker_line_color='navy',
        marker_line_width=1,
        textposition='outside',
        texttemplate='n=%{text}',
        textfont=dict(size=10),
        error_x=dict(color='gray', thickness=1.5)
    )
    return fig


def generate_model_dataset_comparison(config_data_all_models, shots_selected, num_datasets=5):
    """
    Generate box plots comparing models across randomly selected datasets.
    """
    # Prepare data for plotting
    plot_data = []

    # Get all unique datasets
    all_datasets = set()
    for model, data in config_data_all_models.items():
        all_datasets.update(data['dataset'].unique())

    # Randomly select datasets
    selected_datasets = np.random.choice(list(all_datasets), min(num_datasets, len(all_datasets)), replace=False)

    # Prepare data for selected datasets
    for model, data in config_data_all_models.items():
        filtered_data = data[data['dataset'].isin(selected_datasets)]
        for _, row in filtered_data.iterrows():
            plot_data.append({
                'model': model.split('_')[-1],  # Clean model name
                'dataset': row['dataset'],
                'accuracy': row['accuracy']
            })

    plot_df = pd.DataFrame(plot_data)

    # Create a figure for each dataset
    figs = []
    for dataset in selected_datasets:
        # Filter data for the current dataset
        dataset_data = plot_df[plot_df['dataset'] == dataset]

        # Count samples per model
        model_counts = dataset_data['model'].value_counts().to_dict()

        # Create a box plot for the current dataset
        fig = px.box(
            dataset_data,
            x='model',
            y='accuracy',
            title=f'Model Performance Comparison: {dataset}',
            labels={
                'model': 'Model',
                'accuracy': 'Accuracy'
            },
        )

        # Update layout
        fig.update_layout(
            plot_bgcolor='white',
            width=1000,
            height=500,
            title_x=0.5,
            margin=dict(l=50, r=50, t=50, b=150),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                tickangle=45
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                range=[0, 1]
            ),
        )
        for model_name, count in model_counts.items():
            fig.add_annotation(
                x=model_name,
                y=0,  # Adjusted to align at the bottom of the graph
                text=f"n={count}",
                showarrow=False,
                yanchor='top',
                font=dict(size=10)
            )
        figs.append(fig)

    return figs

def main():
    st.title("Model Configuration Analysis Viewer")
    add_description_style()

    # Sidebar for model and dataset selection
    # Shot selection
    # Load available models/datasets
    results_dir = Path(__file__).parents[1] / "results_local"
    shots_models_data = load_shots_models_datasets(base_dir=results_dir)
    if not shots_models_data:
        st.error("No data found in results_local directory.")
        return

    # Shots selection
    selected_shots = st.sidebar.selectbox(
        "Select Number of Shots",
        options=sorted(shots_models_data.keys())
    )
    if not selected_shots:
        return
    # create 2 tabs
    tabs = st.tabs(["Model Performance", "Model Performance Split by Dataset", "Performance Comparison Across Models"])
    # Model selection
    models = sorted(shots_models_data[selected_shots].keys())
    with tabs[0]:
        figs = []
        config_data_all_models = {}
        for model in models:
            config_data = load_data_all_datasets(results_dir, selected_shots, model)
            config_data_all_models[model] = config_data
        fig = generate_model_performance_comparison_all_datasets(
            config_data_all_models=config_data_all_models,
            shots_selected=selected_shots
        )
        st.plotly_chart(fig, use_container_width=True)

    config_data_all_models_by_dataset = {}
    for model in models:
        config_data = load_data(results_dir, selected_shots, model)
        config_data_all_models_by_dataset[model] = config_data
        fig = generate_model_performance_comparison(
            dataset_statistics=config_data,
            model_name=model,
            shots_selected=selected_shots
        )
        figs.append(fig)
    with tabs[1]:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        # Add dataset count selector
        num_datasets = st.slider(
            "Number of random datasets to compare",
            min_value=2,
            max_value=10,
            value=5
        )

        # Generate and show the plot
        figs = generate_model_dataset_comparison(
            config_data_all_models=config_data_all_models_by_dataset,
            shots_selected=selected_shots,
            num_datasets=num_datasets
        )
        # Display each figure separately
        for fig in figs:
            st.plotly_chart(fig, use_container_width=False)


if __name__ == "__main__":
    main()

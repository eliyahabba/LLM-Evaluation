import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")


# ---------------------------------------------------------------------
# Utility functions to load dimension-based data
# ---------------------------------------------------------------------
def load_dimension_data(base_dir, shots, model, dataset, dimension):
    """
    Load dimension-based Parquet data for accuracy analysis (mean/median).
    Assumes the file is located in a subdirectory named after the dimension,
    e.g., {dimension}_analysis.parquet
    """
    model_dir = os.path.join(base_dir, f"Shots_{shots}", model)
    dimension_dir = os.path.join(model_dir, dataset, dimension)

    data_path = os.path.join(dimension_dir, f"prompt_configuration_analyzer.parquet")
    config_data = pd.read_parquet(data_path) if os.path.exists(data_path) else None

    # Load distance matrix (quad-based)
    matrix_path = os.path.join(dimension_dir, "hamming_distance_matrix.parquet")
    distance_matrix = pd.read_parquet(matrix_path) if os.path.exists(matrix_path) else None

    # Load samples data
    samples_path = os.path.join(dimension_dir, "samples_accuracy.parquet")
    samples_data = pd.read_parquet(samples_path) if os.path.exists(samples_path) else None

    return config_data, distance_matrix, samples_data


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------

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


def description_box(title: str, description: str):
    """
    Creates a styled description box with a title and description text.
    """
    st.markdown(f"""
        <div class="graph-description">
            <div class="graph-description-title">{title}</div>
            {description}
    """, unsafe_allow_html=True)


def plot_accuracy_distribution(config_data: pd.DataFrame, x_column="accuracy"):
    """
    Create a histogram showing the distribution of configuration accuracies with improved styling.
    Values are displayed as percentages (0-100).

    Args:
        config_data: DataFrame containing the accuracy data
        x_column: Column name for the accuracy values (default="accuracy")
    """
    # Convert to percentages if needed (multiply by 100 if values are between 0-1)
    if config_data[x_column].max() <= 1:
        data = config_data[x_column] * 100
    else:
        data = config_data[x_column]

    # Create figure and axis with moderate size
    fig, ax = plt.subplots(figsize=(8, 4))

    # Add grid and ensure it's behind the plot
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Create bins from 0 to 100 in steps of 5
    bins = np.arange(0, 101, 5)

    # Plot histogram with customized style
    n, bins, patches = ax.hist(data,
                               bins=bins,
                               color='lightgreen',
                               edgecolor='darkgreen',
                               alpha=0.7)

    # Customize title and labels
    ax.set_title("Distribution of Configuration Accuracies", fontsize=14, pad=15)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Number of Configurations", fontsize=12)

    # Set x-axis limits explicitly
    ax.set_xlim(0, 100)

    # Customize ticks
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{int(i)}' for i in bins], rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add count labels on top of each bar
    for i, count in enumerate(n):
        if count > 0:  # Only add label if bar height > 0
            ax.text(bins[i] + 2.5, count,  # Center of bar
                    f'{int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='darkgreen',
                    fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # For Streamlit display

    return fig, """
This histogram shows the distribution of accuracy scores across different configurations:
- The x-axis shows the accuracy percentage (0% to 100%)
- The y-axis shows how many configurations achieved each accuracy level
- Numbers above bars indicate the exact count of configurations
"""


def plot_accuracy_distribution_dim(config_data: pd.DataFrame, x_column="accuracy"):
    """Create a bar chart of accuracy values with configuration counts."""
    grouped_data = config_data.groupby('quad').agg({
        x_column: 'mean',
        'combination_count': 'first'
    }).reset_index()

    fig = px.bar(
        grouped_data,
        x='quad',
        y=x_column,
        text='combination_count',
        title="Accuracy by Configuration Type"
    )

    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=12, color='black'),
        texttemplate='Configurations: %{text:,}'
    )

    fig.update_layout(
        xaxis_title="Configuration Type",
        yaxis_title="Average Accuracy Score (0-1)",
        title={
            'text': "Accuracy by Configuration Type<br><sup>Each bar shows a configuration's accuracy with the number of variations tested</sup>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=50)
    )
    return fig, """
This bar chart compares the accuracy of different configuration types:
- Each bar represents a unique configuration type
- The height shows the average accuracy (0-1 scale)
- The number above each bar shows how many variations of this configuration were tested
- Higher bars indicate more successful configurations
- The numbers above help understand how thoroughly each configuration was tested
"""


def plot_scatter_distribution(config_data: pd.DataFrame):
    """Create a scatter plot of accuracy vs configuration type with sample counts."""
    fig = px.scatter(
        config_data,
        x="quad",
        y="accuracy",
        text="count",
        labels={
            "quad": "Configuration Type",
            "accuracy": "Accuracy Score (0-1)"
        },
        title="Detailed Configuration Performance"
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05]),
        title={
            'text': "Detailed Configuration Performance<br><sup>Each point shows a configuration's accuracy and number of test samples</sup>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig, """
This scatter plot shows detailed performance for each configuration:
- Each point represents a specific configuration setup
- The y-axis shows the accuracy achieved (0-1 scale)
- The number above each point shows how many test samples were used
- Higher points indicate better performing configurations
"""


def plot_histogram(data: pd.DataFrame, column: str, title: str):
    """
    Plot a histogram of fraction correct with improved styling.

    Args:
        data: DataFrame containing the data
        column: Column name for the data to plot
        title: Title for the histogram
    """
    # Convert fraction to percentage
    data[column] = data[column] * 100

    # Create figure and axis with smaller size
    fig, ax = plt.subplots(figsize=(8, 4))  # Reduced figure size

    # Add grid and ensure it's behind the plot
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Create bins from 0 to 100 in steps of 5
    bins = np.arange(0, 101, 5)  # Changed to stop at 100

    # Plot histogram with customized style
    n, bins, patches = ax.hist(data[column],
                               bins=bins,
                               color='skyblue',
                               edgecolor='black',
                               alpha=0.7)

    # Customize title and labels
    ax.set_title(title, fontsize=14, pad=15)  # Reduced font size
    ax.set_xlabel("Fraction Correct (%)", fontsize=12)
    ax.set_ylabel("Count of Questions", fontsize=12)

    # Set x-axis limits explicitly
    ax.set_xlim(0, 100)

    # Customize ticks
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{int(i)}' for i in bins], rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=10)  # Reduced tick label size

    # Add count labels on top of each bar
    for i, count in enumerate(n):
        if count > 0:  # Only add label if bar height > 0
            ax.text(bins[i] + 2.5, count,  # Center of bar
                    f'{int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=10,  # Reduced font size
                    color='darkblue',
                    fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # For Streamlit display

    # Return description
    return fig, f"""
This histogram shows the distribution of the fraction correct (in percentages) across questions:
- The x-axis shows the percentage of correct predictions (bins of 5%)
- The y-axis shows the number of questions in each bin
- Numbers above bars indicate the exact count in each bin
"""


def plot_square_heatmap(data: pd.DataFrame, axis_name: str, title: str, height=800, width=400):
    """
    Create an interactive heatmap where each cell is a square, and values are displayed in percentages.
    Modified for a taller, narrower display in Streamlit.

    Args:
        data: DataFrame with columns ['dataset', 'sample_index', axis_name, 'fraction_correct'].
        axis_name: The axis to use for configurations (e.g., 'template').
        title: Title for the heatmap.
        height: Height of the heatmap figure (default=800 for taller display).
        width: Width of the heatmap figure (default=400 for narrower display).

    Returns:
        Plotly Heatmap and description.
    """
    # Pivot the data to create a matrix
    heatmap_data = data.pivot(index='sample_index', columns=axis_name, values='fraction_correct') * 100

    # Generate labels for rows and columns
    x_labels = heatmap_data.columns.tolist()
    y_labels = heatmap_data.index.tolist()
    z_values = heatmap_data.values

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        colorbar=dict(
            title="Fraction Correct (%)",
            len=0.9,  # Make colorbar shorter than plot height
            y=0.5,  # Center the colorbar
            yanchor='middle'
        ),
    ))

    # Configure layout for tall, narrow display
    fig.update_layout(
        title={
            'text': f"{title}<br><sup>Each cell represents the fraction correct (%)</sup>",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f"{axis_name.capitalize()} Configurations",
        yaxis_title="Sample Index",
        height=height,
        width=width,
        yaxis=dict(
            autorange='reversed',
            scaleanchor='x',
            scaleratio=1,
            tickmode="array",
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            tickfont=dict(size=10)  # Smaller font for y-axis labels
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            tickangle=45,  # Angle the x-axis labels
            tickfont=dict(size=10),  # Smaller font for x-axis labels
            constrain='domain'
        ),
        margin=dict(l=50, r=50, t=70, b=80)  # Adjusted margins
    )

    # Add description
    description = f"""
    This heatmap visualizes the fraction correct (in percentages) for each sample across different {axis_name} configurations:
    - Each row represents a sample (indexed by sample_index).
    - Each column represents a configuration from the {axis_name} axis.
    - The color scale indicates the fraction correct for that configuration on the sample:
      - Darker colors (e.g., purple) indicate lower accuracy.
      - Brighter colors (e.g., yellow) indicate higher accuracy.
    - The aspect ratio ensures each cell is a square for better readability.
    """
    return fig, description


def plot_distance_heatmap(distance_matrix, threshold=0.2):
    """
    Plot interactive heatmap of configuration pairs based on Hamming distance.

    Args:
        distance_matrix: pandas DataFrame with Hamming distances (0=identical, 1=completely different)
        threshold: distance threshold (default: 0.2)
        show_similar: if True, show pairs with distance below threshold (default: True)
    """
    # Get upper triangle indices and values (excluding diagonal)
    rows, cols = np.triu_indices_from(distance_matrix.values, k=1)
    distances = distance_matrix.values[rows, cols]

    # Filter by threshold (lower distance = more similar)
    mask = distances <= threshold
    filtered_rows = rows[mask]
    filtered_cols = cols[mask]

    if len(filtered_rows) == 0:
        return

    # Get unique indices for filtered pairs
    unique_indices = sorted(set(filtered_rows) | set(filtered_cols))
    filtered_matrix = distance_matrix.iloc[unique_indices, unique_indices]
    return filtered_matrix


def plot_heatmap(distance_matrix: pd.DataFrame, threshold=1., height=1600, width=1600):
    """Create an interactive heatmap showing configuration similarities."""
    distance_matrix = plot_distance_heatmap(distance_matrix, threshold=threshold)
    if distance_matrix is None:
        st.warning("No pairs found matching the criteria")
        return None, ""

    config_labels = distance_matrix.columns.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        x=config_labels,
        y=config_labels,
        colorscale='Viridis',
        showscale=True,
    ))

    fig.update_layout(
        title={
            'text': "Configuration Similarity Matrix<br><sup>Darker colors indicate more similar configurations</sup>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Configuration Types",
        yaxis_title="Configuration Types",
        height=height,
        width=width,
        yaxis={'autorange': 'reversed', 'scaleanchor': 'x', 'scaleratio': 1},
        xaxis={'showticklabels': True, 'tickangle': 45, 'tickfont': {'size': 8}, 'constrain': 'domain'},
        yaxis_tickfont={'size': 16},
        xaxis_tickfont={'size': 16},
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig, """
    This heatmap shows similarity between groups of configurations based on their majority vectors:
 - Each cell shows the Hamming distance between majority vectors of two dimension values
 - The majority vector for each dimension value is created by taking the most common bit (0/1) at each position across all configurations sharing that dimension value
 - Darker colors indicate dimension values whose majority patterns are more similar
 - The diagonal is always darkest (each dimension value's majority pattern is identical to itself)
"""


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


def load_data(base_dir, shots, model, dataset):
    """Load parquet (quad-level accuracy data) and the 'quad' distance matrix for a model/dataset."""
    model_dir = os.path.join(base_dir, f"Shots_{shots}", model)
    dataset_dir = os.path.join(model_dir, dataset)

    # Load parquet data
    parquet_path = os.path.join(dataset_dir, "prompt_configuration_analyzer.parquet")
    config_data = pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else None

    # Load distance matrix (quad-based)
    matrix_path = os.path.join(dataset_dir, "hamming_distance_matrix.parquet")
    distance_matrix = pd.read_parquet(matrix_path) if os.path.exists(matrix_path) else None

    # Load samples data
    samples_path = os.path.join(dataset_dir, "samples_accuracy.parquet")
    samples_data = pd.read_parquet(samples_path) if os.path.exists(samples_path) else None

    return config_data, distance_matrix, samples_data


# ---------------------------------------------------------------------
# Streamlit main
# ---------------------------------------------------------------------
def main():
    st.title("Model Configuration Analysis Viewer")
    add_description_style()

    # Sidebar for model and dataset selection
    st.sidebar.header("Settings")

    # Shot selection
    # Load available models/datasets
    results_dir = Path(__file__).parent / "results_local_without_place_correct_first_last"
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

    # Model selection
    models = sorted(shots_models_data[selected_shots].keys())
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=models
    )
    if not selected_model:
        return

    # Dataset selection
    datasets = sorted(shots_models_data[selected_shots][selected_model])
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=datasets
    )
    if not selected_dataset:
        return

    # Load the "quad"-level data
    config_data, distance_matrix, samples_data = load_data(results_dir, selected_shots, selected_model,
                                                           selected_dataset)

    # Create tabs: "All Data" + 4 dimension tabs
    columns_to_analyze = ["template", "separator", "enumerator", "choices_order"]
    tab_labels = ["All Data"] + [dim.capitalize() for dim in columns_to_analyze]
    tabs = st.tabs(tab_labels)

    # -----------------------------------------------------
    # Tab 0: All Data (quad-level accuracy and distance)
    # -----------------------------------------------------
    with tabs[0]:
        st.header(f"All Data (Quad-Level) - {selected_dataset}")

        if config_data is not None and not config_data.empty:
            st.subheader(f"Accuracy Distribution - {len(config_data)} configurations")
            fig_accuracy, acc_desc = plot_accuracy_distribution(config_data)
            st.plotly_chart(fig_accuracy, use_container_width=False)
            description_box("Understanding the Accuracy Distribution", acc_desc)

            st.subheader("Configuration Details")
            fig_scatter, scatter_desc = plot_scatter_distribution(config_data)
            st.plotly_chart(fig_scatter, use_container_width=True)
            description_box("Understanding the Configuration Details", scatter_desc)
        else:
            st.warning("No quad-level configuration data available.")

        # Quad-level distance matrix
        if distance_matrix is not None and not distance_matrix.empty:
            st.subheader(f"Hamming Distance Matrix - Quad")
            threshold = st.slider("Distance Threshold", min_value=0.0, max_value=1.0, value=1., step=0.05)
            fig_quad_heatmap, heatmap_desc = plot_heatmap(distance_matrix, threshold=threshold)
            if fig_quad_heatmap is not None:
                st.plotly_chart(fig_quad_heatmap, use_container_width=True)
                description_box("Understanding the Hamming Distance Matrix", heatmap_desc)

        else:
            st.warning("No quad-level distance matrix available.")

        if samples_data is not None and not samples_data.empty:
            st.subheader("Samples Accuracy Data")
            fig_dim_samples_hist, dim_samples_hist_desc = plot_histogram(samples_data, "fraction_correct",
                                                                         "Fraction Correct by Sample")
            st.plotly_chart(fig_dim_samples_hist, use_container_width=False)
            description_box(f"Understanding the Samples Histogram", dim_samples_hist_desc)

        # -----------------------------------------------------
        # Tabs 1..4: Each dimension
        # -----------------------------------------------------
    for i, dimension in enumerate(columns_to_analyze, start=1):
        with tabs[i]:
            st.header(f"Dimension: {dimension.title()} - {selected_dataset}")

            config_data, distance_matrix, samples_data = load_dimension_data(results_dir, selected_shots,
                                                                             selected_model, selected_dataset,
                                                                             dimension)
            if config_data is not None and not config_data.empty:
                st.subheader(f"{dimension.title()} Accuracy Data (Mean/Median)")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Mean Accuracy")
                    fig_accuracy, mean_desc = plot_accuracy_distribution_dim(config_data, x_column="mean_accuracy")
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                    description_box(f"Understanding Mean Accuracy for {dimension.title()}", mean_desc)

                with col2:
                    st.subheader("Median Accuracy")
                    fig_accuracy, median_desc = plot_accuracy_distribution_dim(config_data, x_column="median_accuracy")
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                    description_box(f"Understanding Median Accuracy for {dimension.title()}", median_desc)
            else:
                st.warning(f"No {dimension.title()} dimension accuracy data available.")

            if distance_matrix is not None and not distance_matrix.empty:
                st.subheader(f"Hamming Distance Matrix - {dimension.title()}")
                fig_dim_heatmap, dim_heatmap_desc = plot_heatmap(distance_matrix, height=800, width=800)
                if fig_dim_heatmap is not None:
                    st.plotly_chart(fig_dim_heatmap, use_container_width=True)
                    description_box(f"Understanding the {dimension.title()} Hamming Distance Matrix", dim_heatmap_desc)
            else:
                st.warning(f"No {dimension.title()} dimension distance matrix available.")

            if samples_data is not None and not samples_data.empty:
                st.subheader(f"Samples Accuracy Data - {dimension.title()}")
                fig_dim_samples_hist, dim_samples_hist_desc = plot_square_heatmap(samples_data, dimension,
                                                                                  title="Square Heatmap of Fraction Correct by Template"
                                                                                  )
                st.plotly_chart(fig_dim_samples_hist, use_container_width=False)
                description_box(f"Understanding the {dimension.title()} Samples Histogram", dim_samples_hist_desc)


if __name__ == "__main__":
    main()

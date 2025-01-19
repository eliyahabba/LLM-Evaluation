import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    data_path = os.path.join(dimension_dir, f"{dimension}_analysis.parquet")

    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    return None


def load_dimension_distance_matrix(base_dir, shots, model, dataset, dimension):
    """
    Load the dimension-based Hamming distance matrix Parquet file.
    Assumes it's located under:
       {base_dir}/Shots_{shots}/{model}/{dataset}/{dimension}/hamming_distance_matrix_{dimension}.parquet
    """
    model_dir = os.path.join(base_dir, f"Shots_{shots}", model)
    dimension_dir = os.path.join(model_dir, dataset, dimension)
    dist_path = os.path.join(dimension_dir, f"hamming_distance_matrix_{dimension}.parquet")

    if os.path.exists(dist_path):
        return pd.read_parquet(dist_path)
    return None


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------
def plot_accuracy_distribution(config_data: pd.DataFrame, x_column="accuracy"):
    """Create and return a histogram of 'accuracy' values."""
    fig = px.histogram(
        config_data,
        x=x_column,
        nbins=30,
        title="Accuracy Distribution"
    )
    fig.update_layout(
        xaxis_title="Accuracy",
        yaxis_title="Count"
    )
    return fig


def plot_accuracy_distribution2(config_data: pd.DataFrame, x_column="accuracy"):
    """Create and return a histogram of 'accuracy' values with count annotations."""

    # מחשב ממוצע accuracy לכל קבוצת quad
    grouped_data = config_data.groupby('quad').agg({
        x_column: 'mean',
        'combination_count': 'first'  # מניח שהערך זהה לכל quad
    }).reset_index()

    fig = px.bar(
        grouped_data,
        x='quad',
        y=x_column,
        text='combination_count',  # הערכים שיוצגו מעל העמודות
        title="Accuracy Distribution"
    )

    fig.update_traces(
        textposition='outside',  # מיקום הטקסט מעל העמודות
        cliponaxis=False  # מאפשר לטקסט לחרוג מגבולות הציר
    )

    fig.update_layout(
        xaxis_title="Configuration",
        yaxis_title="Accuracy",
        # מגדיל את המרווח למעלה כדי שיהיה מקום לטקסט
        margin=dict(t=50)
    )


    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=12, color='black'),  # גודל וצבע הטקסט
        texttemplate = 'Configurations: %{text:,}'  # או בעברית: 'דגימות: %{text:,}'
)
    return fig


def plot_scatter_distribution(config_data: pd.DataFrame):
    """
    Create a scatter plot of 'accuracy' vs. 'quad', labeling points with 'count'.
    """
    fig = px.scatter(
        config_data,
        x="quad",
        y="accuracy",
        text="count",
        labels={"quad": "Configuration", "accuracy": "Accuracy"}
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05])
    )
    return fig


def plot_dimension_scatter(dimension_data: pd.DataFrame, y_column: str, focus_dimension: str):
    """
    Create a scatter plot for dimension-based data, plotting 'quad' vs. y_column.
    We label each point with 'combination_count' (how many combos contributed).
    """
    fig = px.scatter(
        dimension_data,
        x="quad",
        y=y_column,
        text="combination_count",
        title=f"{y_column.replace('_', ' ').title()} - {focus_dimension}",
        labels={"quad": f"{focus_dimension} (others = ALL)", y_column: y_column.replace('_', ' ').title()}
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05])
    )
    return fig


def plot_heatmap(distance_matrix: pd.DataFrame, height=1600, width=1600):
    """
    Create an interactive heatmap using plotly for the given distance_matrix DataFrame.
    The row/column labels are taken from distance_matrix.index/columns.
    """
    config_labels = distance_matrix.columns.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        x=config_labels,
        y=config_labels,
        colorscale='Viridis',
        showscale=True,
    ))

    fig.update_layout(
        title="Hamming Distance Matrix",
        xaxis_title="Configurations",
        yaxis_title="Configurations",
        height=height,
        width=width,
        yaxis={'autorange': 'reversed', 'scaleanchor': 'x', 'scaleratio': 1},
        xaxis={'showticklabels': True, 'tickangle': 45, 'tickfont': {'size': 8}, 'constrain': 'domain'},
        yaxis_tickfont={'size': 16},
        xaxis_tickfont={'size': 16},
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


# ---------------------------------------------------------------------
# Loaders for model, dataset, overall data
# ---------------------------------------------------------------------
def load_model_datasets(base_dir="results_local"):
    """Load available models and their datasets from the given base_dir."""
    models = {}
    if not os.path.isdir(base_dir):
        return models

    for shots_dir in os.listdir(base_dir):
        if not shots_dir.startswith("Shots_"):
            continue
        shots_path = os.path.join(base_dir, shots_dir)
        if not os.path.isdir(shots_path):
            continue

        for model in os.listdir(shots_path):
            model_path = os.path.join(shots_path, model)
            if not os.path.isdir(model_path):
                continue

            # Initialize if new model
            if model not in models:
                models[model] = set()

            # Add datasets for this model
            for dataset in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset)
                if os.path.isdir(dataset_path):
                    models[model].add(dataset)

    return models


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
    return config_data, distance_matrix


# ---------------------------------------------------------------------
# Streamlit main
# ---------------------------------------------------------------------
def main():
    st.title("Model Configuration Analysis Viewer")

    # Sidebar for model and dataset selection
    st.sidebar.header("Settings")

    # Shot selection
    shots_options = [0, 5]
    selected_shots = st.sidebar.selectbox("Select Shots", shots_options)

    # Load available models/datasets
    models_data = load_model_datasets(base_dir="results_local")

    if not models_data:
        st.error("No data found in results_local directory.")
        return

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=sorted(models_data.keys())
    )
    if not selected_model:
        return

    # Dataset selection
    datasets = sorted(models_data[selected_model])
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=datasets
    )
    if not selected_dataset:
        return

    # Load the "quad"-level data
    config_data, distance_matrix = load_data("results_local", selected_shots, selected_model, selected_dataset)

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
            fig_accuracy = plot_accuracy_distribution(config_data)
            st.plotly_chart(fig_accuracy, use_container_width=True)

            st.subheader("Configuration Details")
            fig_scatter = plot_scatter_distribution(config_data)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No quad-level configuration data available.")

        # Quad-level distance matrix
        if distance_matrix is not None and not distance_matrix.empty:
            st.subheader(f"Hamming Distance Matrix - Quad")
            fig_quad_heatmap = plot_heatmap(distance_matrix)
            st.plotly_chart(fig_quad_heatmap, use_container_width=True)
        else:
            st.warning("No quad-level distance matrix available.")

    # -----------------------------------------------------
    # Tabs 1..4: Each dimension
    # -----------------------------------------------------
    for i, dimension in enumerate(columns_to_analyze, start=1):
        with tabs[i]:
            st.header(f"Dimension: {dimension.title()} - {selected_dataset}")

            # 1) Load dimension-based accuracy data
            dimension_data = load_dimension_data("results_local", selected_shots, selected_model, selected_dataset, dimension)
            if dimension_data is not None and not dimension_data.empty:
                st.subheader(f"{dimension.title()} Accuracy Data (Mean/Median)")

                # Show mean accuracy on the left, median accuracy on the right
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Mean Accuracy")
                    # fig_mean = plot_dimension_scatter(dimension_data, "mean_accuracy", dimension)
                    # st.plotly_chart(fig_mean, use_container_width=True)
                    fig_accuracy = plot_accuracy_distribution2(dimension_data, x_column="mean_accuracy")
                    st.plotly_chart(fig_accuracy, use_container_width=True)

                with col2:
                    st.subheader("Median Accuracy")
                    # fig_median = plot_dimension_scatter(dimension_data, "median_accuracy", dimension)
                    # st.plotly_chart(fig_median, use_container_width=True)
                    fig_accuracy = plot_accuracy_distribution2(dimension_data, x_column="median_accuracy")
                    st.plotly_chart(fig_accuracy, use_container_width=True)
            else:
                st.warning(f"No {dimension.title()} dimension accuracy data available.")

            # 2) Load dimension-based distance matrix
            dim_dist_matrix = load_dimension_distance_matrix("results_local", selected_shots, selected_model, selected_dataset, dimension)
            if dim_dist_matrix is not None and not dim_dist_matrix.empty:
                st.subheader(f"Hamming Distance Matrix - {dimension.title()}")
                fig_dim_heatmap = plot_heatmap(dim_dist_matrix, height=800, width=800)
                st.plotly_chart(fig_dim_heatmap, use_container_width=True)
            else:
                st.warning(f"No {dimension.title()} dimension distance matrix available.")

if __name__ == "__main__":
    main()
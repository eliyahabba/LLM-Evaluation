import sys
from pathlib import Path

import streamlit as st

file_path = Path(__file__).parents[4]
sys.path.append(str(file_path))
from src.analysis.create_plots.DataLoader import DataLoader


# Make sure your data_loader.py is accessible (e.g., same folder or a proper Python package).
# from data_loader import DataLoader

# A placeholder for your DataLoader class if needed;
# you can remove this if you actually import from data_loader.

def load_data(model_name=None, shots=None, max_samples=None):
    """
    Loads the dataset from Hugging Face, filtering by model_name and shots if provided.
    Leaves 'generated_text' and 'ground_truth' intact (not dropping them).
    Returns a pandas DataFrame.
    """
    # In an actual usage, you'd do:
    # dataset = load_dataset(self.dataset_name, split=self.split, max_samples=max_samples)
    # For brevity, let's assume you fetch the data as a DataFrame directly:
    # Here is a trivial example (replace with real huggingface logic).

    data_loader = DataLoader()
    df_partial = data_loader.load_and_process_data(model_name, shots=shots,
                                                   max_samples=max_samples, drop=False)

    return df_partial


# -------------- CACHED LOADING --------------
@st.cache_data(show_spinner=True)
def load_entire_dataset(model_name=None, shots=None, max_samples=None):
    """
    Loads the dataset from Hugging Face (or local), possibly filtered by model/shots,
    and returns a DataFrame. Caches the result to avoid re-loading repeatedly.
    """
    df = load_data(model_name=model_name, shots=shots, max_samples=max_samples)
    return df


def main():
    st.set_page_config(page_title="Multi-Filter Viewer", layout="wide")
    st.title("LLM Multi-Filter Viewer")

    # --------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------
    st.sidebar.header("Global Settings")
    limit_samples = st.sidebar.checkbox("Limit number of samples", value=False)
    max_samples = None
    if limit_samples:
        max_samples = st.sidebar.number_input(
            "Max samples to load",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )

    # We can optionally load by a single model/shots upfront to reduce data size in memory
    # (If you prefer to load the entire dataset at once, just remove these 2 fields.)
    shots_to_evaluate = [0, 5]
    models_to_evaluate = [
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    st.sidebar.markdown('Recommended to load a specific model and shots initially to reduce data size.')
    selected_model_for_loading = st.sidebar.selectbox("Model to load initially (optional)", models_to_evaluate, index=1)
    selected_shots_for_loading = st.sidebar.selectbox("Shots to load initially (optional)",
                                                      [s for s in shots_to_evaluate], index=1)

    st.sidebar.markdown("---")
    st.sidebar.header("Filter on the DataFrame Columns")

    # We won't load all unique options from the dataset until after we've loaded it,
    # so let's do the load now.
    # We do a button to confirm load, or automatically load. Let's do automatic load:
    with st.spinner("Loading data..."):
        df = load_entire_dataset(
            model_name=selected_model_for_loading if selected_model_for_loading else None,
            shots=selected_shots_for_loading if selected_shots_for_loading else None,
            max_samples=max_samples
        )

    if df.empty:
        st.error("No data loaded. Try different loading filters or check your dataset.")
        return

    # Now that we have some data in `df`, let's gather the unique values for each column
    # for the final filtering.
    # The user specifically wants 7 columns:
    columns_to_filter = ['dataset', 'template', 'separator', 'enumerator', 'choices_order']

    filter_values = {}
    for col in columns_to_filter:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        # Insert a "None" or "All" option at the start
        filter_select = st.sidebar.selectbox(
            f"Filter by {col}",
            options=["(All)"] + unique_vals,
            key=f"filter_{col}"
        )
        if filter_select != "(All)":
            filter_values[col] = filter_select  # means we want to filter on this value

    # Also, how many rows to show at once:
    display_limit = st.sidebar.number_input("Max rows to display", min_value=1, max_value=2000, value=50)

    # --------------------------------
    # APPLY FILTERS
    # --------------------------------
    df_filtered = df.copy()
    for col, val in filter_values.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    st.write(f"### Filtered Results (showing up to {display_limit} rows)")
    st.write(f"Found {len(df_filtered)} matching rows.")

    if len(df_filtered) == 0:
        st.warning("No matching rows for the selected filters.")
        return

    # Slice to limit the displayed rows
    df_display = df_filtered.head(display_limit)

    # ------------------------------------------------
    # DISPLAY ROWS ONE-BY-ONE
    # ------------------------------------------------
    # Because the user said "display them one after the other with generated_text and ground_truth"
    # We'll show each row in an expander or something similar:
    for idx, row in df_display.iterrows():
        with st.expander(f"Row index: {idx} | {row['model']} / {row['dataset']} / shots={row['shots']}"):
            st.markdown(f"**Template**: `{row['template']}`")
            st.markdown(f"**Separator**: `{row['separator']}`")
            st.markdown(f"**Enumerator**: `{row['enumerator']}`")
            st.markdown(f"**Choices Order**: `{row['choices_order']}`")

            # Show text fields
            st.write("**Generated Text:**")
            st.text(row.get('generated_text', 'N/A'))

            st.write("**Ground Truth:**")
            st.text(row.get('ground_truth', 'N/A'))

            # If there's a score or sample_index, we can show them:
            score = row.get('score', None)
            if score is not None:
                st.write(f"Score: {score}")

            sample_idx = row.get('sample_index', None)
            if sample_idx is not None:
                st.write(f"Sample Index: {sample_idx}")

    # Optionally, also show a DataFrame table at the bottom
    with st.expander("See Filtered Data as a Table"):
        st.dataframe(df_display)


if __name__ == "__main__":
    main()

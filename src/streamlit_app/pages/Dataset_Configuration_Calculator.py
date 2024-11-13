import itertools

import pandas as pd
import streamlit as st
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams


def calculate_combinations(dataset_configs, selected_datasets, selected_models, selected_quant,
                           selected_prompts_variations,
                           selected_prompt_paraphrasing,
                           samples_per_dataset,
                           shots,
                           additional_features
                           ):
    # Calculate all combinations using itertools.product
    expanded_datasets = []
    for ds, num_sub_ds in dataset_configs.items():
        if num_sub_ds > 1:
            expanded_datasets.extend([f"{ds}-{i + 1}" for i in range(num_sub_ds)])
        else:
            expanded_datasets.append(ds)
    # take only the values with positive (additional_features van be empty)
    # Handle empty additional_features
    if not additional_features:
        additional_features = [None]  # Use None as a placeholder for no additional features

    combinations = list(itertools.product(
        expanded_datasets,
        selected_models,
        selected_quant,
        range(1, selected_prompts_variations + 1),
        range(1, selected_prompt_paraphrasing + 1),
        shots,
        additional_features
    ))

    # Calculate total samples and estimated cost
    total_combinations = len(combinations)
    total_samples = total_combinations * samples_per_dataset
    estimated_cost = total_samples * 0.01  # $0.01 per instance as per document

    return combinations, total_combinations, total_samples, estimated_cost


def main():
    st.set_page_config(page_title="Dataset Configuration Calculator", layout="wide")
    PROMPT_OPTIONS = ConfigParams.override_options
    GREEK_CHARS = "αβγδεζηθικ"  # 10 Greek letters
    RARE_CHARS = "œ§Жüϡʘϗѯ₿⊗"  # 10 rare characters
    PROMPT_OPTIONS['enumerator'].extend([GREEK_CHARS, RARE_CHARS])

    st.title("Dataset Configuration Calculator")

    # Define available options
    # Update the datasets dictionary to include sub-dataset information
    datasets = {
        'MMLU': {'total_instances': 14042, 'sub_datasets': 57},
        'MMLU-Pro': {'total_instances': 12032, 'sub_datasets': 14},
        'ARC-Challenge': {'total_instances': 1172, 'sub_datasets': 0},
        'Bool Q': {'total_instances': 3270, 'sub_datasets': 0},
        'HellaSwag': {'total_instances': 6700, 'sub_datasets': 0},
        'Social IQA': {'total_instances': 1954, 'sub_datasets': 0},
        'GPQA': {'total_instances': 448, 'sub_datasets': 0}
    }

    # Replace the dataset selection section in the main() function with:
    models = [
        "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
        "Llama-3-8B-Instruct",
        "Llama-3-70B-Instruct",
        'Llama2-7B-Instruct', 'Llama2-13B-Instruct', 'Llama2-70B',
        'Mistral-V1', 'Mistral-V2', 'Mistral-V3',
        'Mixtral',
        'Gemma-2B-Instruct', 'Gemma-7B-Instruct'
    ]
    default_models = [model for model in models if "Llama" in model]
    default_models = models[:1]
    quantizations = ["None", '4int', '8int']
    default_quant = ["None"]
    shots = ["zero_shot", "tow_shot", "four_shot"]
    default_shots = ["zero_shot"]
    Additional_features = ["None",
                           "Spelling_errors", "Random spaces"]
    default_features = ["None"]

    # Create columns for different parameter selections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Parameters")

        # Dataset selection
        selected_datasets = st.multiselect(
            "Select Datasets",
            options=list(datasets.keys()),
            help="Choose one or more datasets",
            default=["MMLU", "MMLU-Pro"]
        )

        # Dataset selection
        # Sub-dataset selection for each selected dataset
        dataset_configs = {}
        if selected_datasets:
            st.write("Configure sub-datasets:")
            for ds in selected_datasets:
                if datasets[ds]['sub_datasets'] > 0:
                    col_name, col_slider = st.columns([2, 3])
                    with col_name:
                        st.write(f"{ds}:")
                    with col_slider:
                        num_sub_datasets = st.slider(
                            f"Number of sub-datasets for {ds}",
                            min_value=1,
                            max_value=datasets[ds]['sub_datasets'],
                            value=datasets[ds]['sub_datasets'],  # Default to max
                            key=f"slider_{ds}"
                        )
                    dataset_configs[ds] = num_sub_datasets
                else:
                    dataset_configs[ds] = 1
                    st.write(f"{ds}: No sub-datasets")

        # Show selected dataset sizes
        if selected_datasets:
            st.write("Selected Dataset Sizes:")
            for ds in selected_datasets:
                st.write(f"- {ds}: {datasets[ds]} instances")

        # Model selection
        selected_models = st.multiselect(
            "Select Models",
            options=models,
            help="Choose one or more models",
            default=default_models
        )

        # Quantization selection
        selected_quant = st.multiselect(
            "Select Quantizations",
            options=quantizations,
            help="Choose one or more quantization options",
            default=default_quant
        )

        # Number of prompt variations
        num_prompt_paraphrasing = st.slider(
            "Number of Prompt Paraphrasing",
            min_value=1,
            max_value=10,
            value=2,
            help="Select number of prompt paraphrasing"
        )
        st.subheader("Prompt Variations")

        with st.expander("Configure Prompt Variations", expanded=True):
            selected_enumerators = st.multiselect(
                "Select Enumerators",
                options=PROMPT_OPTIONS["enumerator"],
                default=PROMPT_OPTIONS["enumerator"],
                help="How to enumerate choices (e.g., A,B,C or 1,2,3)"
            )

            selected_separators = st.multiselect(
                "Select Choice Separators",
                options=PROMPT_OPTIONS["choices_separator"],
                default=PROMPT_OPTIONS["choices_separator"],
                help="How to separate choices"
            )
            selected_shuffle = st.multiselect(
                "Shuffle Choices",
                options=["Yes", "No"],
                default=["Yes", "No"],
                help="Whether to shuffle the order of choices"
            )
            # Convert Yes/No to boolean
            selected_shuffle = [s == "Yes" for s in selected_shuffle]

            # Calculate total variations
            num_prompt_variations = (
                    len(selected_enumerators) *
                    len(selected_separators) *
                    len(selected_shuffle)
            )

            st.metric("Total Prompt Variations", num_prompt_variations)

        # Samples per dataset
        samples_per_dataset = st.number_input(
            "Samples per Dataset",
            min_value=100,
            max_value=10000,
            value=100,
            step=100,
            help="Number of samples to use from each dataset"
        )

        shots = st.multiselect(
            "Select Shots",
            options=shots,
            help="Choose one or more shots",
            default=default_shots
        )

        additional_features = st.multiselect(
            "Select Additional Features",
            options=Additional_features,
            help="Choose one or more features",
            default=default_features
        )

    with col2:
        st.subheader("Results")

        if all([selected_datasets, selected_models, selected_quant]):
            combinations, total_combinations, total_samples, estimated_cost = calculate_combinations(dataset_configs,
                                                                                                     selected_datasets,
                                                                                                     selected_models,
                                                                                                     selected_quant,
                                                                                                     num_prompt_variations,
                                                                                                     num_prompt_paraphrasing,
                                                                                                     samples_per_dataset,
                                                                                                     shots,
                                                                                                     additional_features
                                                                                                     )

            # Display metrics
            col_metrics1, col_metrics2 = st.columns(2)

            with col_metrics1:
                st.metric("Total Combinations", f"{total_combinations:,}")
                st.metric("Total Samples", f"{total_samples:,}")

            with col_metrics2:
                st.metric("Configurations", len(combinations))

            # Show sample of combinations
            st.subheader("Sample Configurations")
            df = pd.DataFrame(combinations[:1000], columns=['Dataset', 'Model', 'Quantization', 'Prompt Variation',
                                                            ''
                                                            'ng', 'Shots', 'Additional Features'])
            st.dataframe(df, height=400)

            # Download button for full configuration
            if len(combinations) > 0:
                df_full = pd.DataFrame(combinations, columns=['Dataset', 'Model', 'Quantization', 'Prompt Variation',
                                                              'Prompt Paraphrasing', 'Shots', 'Additional Features'])
                csv = df_full.to_csv(index=False)
                st.download_button(
                    label="Download Full Configuration",
                    data=csv,
                    file_name="dataset_configurations.csv",
                    mime="text/csv"
                )
        else:
            st.info("Please select at least one option from each category to see the results.")


if __name__ == "__main__":
    main()

import itertools
import json

import streamlit as st

from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.experiments.experiment_preparation.datasets_configurations.DatasetConfigFactory import DatasetConfigFactory



def calculate_combinations(dataset_configs, selected_datasets, selected_models, selected_quant,
                           selected_prompts_variations,
                           selected_phrases,
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
        *[v for v in selected_prompts_variations.values()],
        selected_phrases,
        shots,
        additional_features
    ))

    # Calculate total samples and estimated cost
    total_combinations = len(combinations)
    total_samples = total_combinations * samples_per_dataset
    estimated_cost = total_samples * 0.01  # $0.01 per instance as per document

    summary = create_experiment_params_summary(
        dataset_configs,
        selected_models,
        selected_quant,
        selected_prompts_variations,
        selected_phrases,
        shots,
        additional_features
    )
    display_summary(summary)

    return combinations, total_combinations, total_samples, estimated_cost


def get_prompt_paraphrasing(selected_datasets):
    dataset_map = {'MMLU': "MMLU",
                   'MMLU-Pro': "MMLU_PRO",
                   'ARC-Challenge': 'AI2_ARC',
                   'ARC-EASY': 'AI2_ARC',
                   'HellaSwag': 'HellaSwag',
                   'OpenBookQA': 'OpenBookQA',
                   'Social-IQa': 'Social_IQa',
                   'race high (Reading Comprehension)': 'RACE',
                   'race middle (Reading Comprehension)': "RACE",
                   'Quality (long global context questions)': 'Quality',
                   'Coursera (long  context about big data and machine learning))': 'Coursera',
                   'TPO (long context machine comprehension of spoken content)': 'TPO',
                   'mgsm (Multilingual of GSM)': 'mgsm'}
    dataset_names_to_prompts_instruct = {}
    for dataset in selected_datasets:
        dataset_names_to_prompts_instruct[dataset] = DatasetConfigFactory.get_instruct_prompts(dataset_map[dataset])
    return dataset_names_to_prompts_instruct


def main():
    st.set_page_config(page_title="Dataset Configuration Calculator", layout="wide")
    if "prompt_variations" not in st.session_state:
        from copy import deepcopy
        st.session_state.prompt_variations = deepcopy(ConfigParams.override_options)
        st.session_state.prompt_variations['choices_separator'] = [item.replace(" ", "\\s") if item == " " else item for
                                                                   item in st.session_state.prompt_variations[
                                                                       'choices_separator']]
        st.session_state.prompt_variations['choices_separator'] = [item.replace("\n", "\\n") for item in
                                                                   st.session_state.prompt_variations[
                                                                       'choices_separator']]

    st.title("Dataset Configuration Calculator")

    # Define available options
    # Update the datasets dictionary to include sub-dataset information
    grouped_datasets = {
        'knowledge_based_reasoning': {
            'MMLU': {'total_instances': 14042, 'sub_datasets': 57},
            'MMLU-Pro': {'total_instances': 12032, 'sub_datasets': 14},
            'ARC-Challenge': {'total_instances': 1172, 'sub_datasets': 0},
            'ARC-EASY': {'total_instances': 1172, 'sub_datasets': 0},
            'HellaSwag': {'total_instances': 6700, 'sub_datasets': 0},
            'OpenBookQA': {'total_instances': 1000, 'sub_datasets': 0},
            'Social-IQa': {'total_instances': 1000, 'sub_datasets': 0}
        },
        'context': {
            'race high (Reading Comprehension)': {'total_instances': 1000, 'sub_datasets': 0},
            'race middle (Reading Comprehension)': {'total_instances': 1000, 'sub_datasets': 0},
            'Quality (long global context questions)': {'total_instances': 1000, 'sub_datasets': 0},
            'Coursera (long  context about big data and machine learning))': {'total_instances': 1000,
                                                                              'sub_datasets': 0},
            'TPO (long context machine comprehension of spoken content)': {'total_instances': 1000, 'sub_datasets': 0}
        },
        'multilingual': {
            'mgsm (Multilingual of GSM)': {'total_instances': 1000, 'sub_datasets': 11}
        }
    }

    flattened_datasets = {}
    for group, datasets in grouped_datasets.items():
        for dataset_name, dataset_info in datasets.items():
            flattened_datasets[dataset_name] = dataset_info

    datasets_to_groups = {}
    for group, datasets in grouped_datasets.items():
        for dataset_name, dataset_info in datasets.items():
            datasets_to_groups[dataset_name] = group
    # Replace the dataset selection section in the main() function with:

    models = ["Llama-3.2-1B-Instruct",
              "Llama-3.2-3B-Instruct",
              "Meta-Llama-3-8B-Instruct",
              "Meta-Llama-3-70B-Instruct",
              "Mistral-7B-Instruct-v0.3",
              "Mixtral-8x7B-Instruct-v0.1",
              "Mixtral-8x22B-Instruct-v0.1",
              "OLMo-7B-Instruct",
              "OLMoE-1B-7B-0924-Instruct",
              "Qwen2.5-0.5B-Instruct",
              "Qwen2.5-3B-Instruct",
              "Qwen2.5-7B-Instruct",
              "Qwen2.5-72B-Instruct"]
    default_models = [
        "Meta-Llama-3-8B-Instruct",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "OLMoE-1B-7B-0924-Instruct",
        "Mistral-7B-Instruct-v0.3"
    ]
    quantizations = ["None", '4int', '8int']
    default_quant = ["None", '8int']
    shots = ["zero_shot", "five_shot"]
    default_shots = ["zero_shot", "five_shot"]
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
            options=list(flattened_datasets.keys()),
            help="Choose one or more datasets",
            default=flattened_datasets.keys()
        )

        # Dataset selection
        # Sub-dataset selection for each selected dataset
        dataset_configs = {}
        if selected_datasets:
            st.write("Configure sub-datasets:")
            for ds in selected_datasets:
                if flattened_datasets[ds]['sub_datasets'] > 0:
                    col_name, col_slider = st.columns([2, 3])
                    with col_name:
                        st.write(f"{ds}:")
                    with col_slider:
                        num_sub_datasets = st.slider(
                            f"Number of sub-datasets for {ds}",
                            min_value=1,
                            max_value=flattened_datasets[ds]['sub_datasets'],
                            value=flattened_datasets[ds]['sub_datasets'],  # Default to max
                            key=f"slider_{ds}"
                        )
                    dataset_configs[ds] = num_sub_datasets
                else:
                    dataset_configs[ds] = 1
                    # st.write(f"{ds}: No sub-datasets")

        # Show selected dataset sizes
        # st.write("Selected Dataset Sizes:")
        # for ds in selected_datasets:
        #     st.write(f"- {ds}: {datasets[ds]} instances")
        if selected_datasets:
            st.write("Selected Dataset:")
            tab1, tab2, tab3 = st.tabs(["knowledge_based_reasoning", "context", "multilingual"])
            for i, item in enumerate(selected_datasets):
                if datasets_to_groups[item] == "knowledge_based_reasoning":
                    with tab1:
                        st.write(f"• {item}")
                elif datasets_to_groups[item] == "context":
                    with tab2:
                        st.write(f"• {item}")
                elif datasets_to_groups[item] == "multilingual":
                    with tab3:
                        st.write(f"• {item}")

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

        # Number of prompt prompt_phrases
        # display the prompt phrases and selcet values from them
        st.info("Select Prompt Phrases- Use the arrows to navigate between datasets", icon="ℹ️")
        prompt_phrases = get_prompt_paraphrasing(selected_datasets)
        tabs = st.tabs(selected_datasets)
        selected_datasets_phrases = {}
        for tab, dataset in zip(tabs, selected_datasets):
            with tab:
                if dataset in prompt_phrases:
                    prompts = prompt_phrases[dataset].get_all_prompts()

                    # Format prompts for display
                    formatted_prompts = {
                        f"{i + 1}. {prompt.name}": prompt.text
                        for i, prompt in enumerate(prompts)
                    }

                    # Create multiselect for this dataset's prompts
                    selected = st.multiselect(
                        f"Select Prompts for {dataset}",
                        options=formatted_prompts.keys(),
                        default=list(formatted_prompts.keys()),
                        key=f"multiselect_{dataset}"  # Unique key for each multiselect
                    )
                    selected_datasets_phrases[dataset] = selected
                    # Show preview expander
                    with st.expander(f"Preview Prompts for {dataset}"):
                        template = st.selectbox(
                            "Select template to view",
                            options=formatted_prompts.keys(),
                            # format_func=lambda x: x.split('. ')[1],  # Remove numbering
                            key=f"selectbox_{dataset}"  # Unique key for each selectbox
                        )
                        if template:
                            st.code(formatted_prompts[template], language="text")

                    # Show count of prompts
                    st.info(f"Number of prompts for {dataset}: {len(prompts)}")
                else:
                    st.warning(f"No prompts available for {dataset}")

        # Get the actual values of selected prompts (if you need them later)
        selected_phrases = selected_datasets_phrases[selected_datasets[0]]
        st.subheader("Prompt Variations")

        with st.expander("Configure Prompt Variations", expanded=True):
            selected_enumerators = st.multiselect(
                "Select Enumerators",
                options=st.session_state.prompt_variations["enumerator"],
                default=st.session_state.prompt_variations["enumerator"],
                help="How to enumerate choices (e.g., A,B,C or 1,2,3)"
            )

            selected_separators = st.multiselect(
                "Select Choice Separators",
                options=st.session_state.prompt_variations["choices_separator"],
                default=st.session_state.prompt_variations["choices_separator"],
                help="How to separate choices"
            )
            selected_shuffle = st.multiselect(
                "Shuffle Choices",
                options=st.session_state.prompt_variations["shuffle_choices"],
                default=st.session_state.prompt_variations["shuffle_choices"],
                help="Whether to shuffle the order of choices"
            )

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
            prompt_variations_all = {
                "enumerator": selected_enumerators,
                "choices_separator": selected_separators,
                "shuffle_choices": selected_shuffle
            }
            combinations, total_combinations, total_samples, estimated_cost = calculate_combinations(dataset_configs,
                                                                                                     selected_datasets,
                                                                                                     selected_models,
                                                                                                     selected_quant,
                                                                                                     prompt_variations_all,
                                                                                                     selected_phrases,
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
            # st.subheader("Sample Configurations")
            # df = pd.DataFrame(combinations[:1000], columns=['Dataset', 'Model', 'Quantization', 'Prompt Variation',
            #                                                 ''
            #                                                 'ng', 'Shots', 'Additional Features'])
            # st.dataframe(df, height=400)

            # Download button for full configuration
            # if len(combinations) > 0:
            #     df_full = pd.DataFrame(combinations, columns=['Dataset', 'Model', 'Quantization', 'Prompt Variation',
            #                                                   'Prompt Paraphrasing', 'Shots', 'Additional Features'])
            #     csv = df_full.to_csv(index=False)
            #     st.download_button(
            #         label="Download Full Configuration",
            #         data=csv,
            #         file_name="dataset_configurations.csv",
            #         mime="text/csv"
            #     )
        else:
            st.info("Please select at least one option from each category to see the results.")


def create_experiment_params_summary(
        dataset_configs,
        selected_models,
        selected_quant,
        selected_prompts_variations,
        selected_phrases,
        shots,
        additional_features
):
    summary = {}

    # Datasets summary
    summary['datasets'] = {
        dataset: f"{count} sub-datasets" if count > 1 else "No sub-datasets"
        for dataset, count in dataset_configs.items()
    }

    # Models summary
    summary['models'] = selected_models

    # Quantization summary
    summary['quantization'] = selected_quant

    # Prompt variations summary
    summary['prompt_variations'] = {
        'total_variations': selected_prompts_variations,
        'selected_phrases': [phrase.split('. ')[1] for phrase in selected_phrases]  # Remove the numbering
    }

    # Shots summary
    summary['shots'] = shots

    # Additional features summary
    summary['additional_features'] = additional_features if additional_features else ['None']

    return summary


def display_summary(summary):
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Parameters Summary",
        data=summary_json,
        file_name="experiment_parameters_summary.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main()

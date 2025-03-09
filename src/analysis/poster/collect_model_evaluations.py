import json
import traceback
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from src.analysis.create_plots.DataLoader import DataLoader

def process_configuration(params):
    """
    Processes a single configuration (model, shots, dataset).
    Loads, filters, and computes accuracy.
    Returns the processed DataFrame along with row counts.
    """
    model_name, shots_selected, dataset = params
    print(f"Processing model: {model_name} with {shots_selected} shots on dataset: {dataset}")

    data_loader = DataLoader()

    try:
        df_partial = data_loader.load_and_process_data(
            model_name=model_name,
            shots=shots_selected,
            datasets=[dataset],
            max_samples=None
        )
    except Exception as e:
        print(f"Warning: Failed to load dataset '{dataset}' for model '{model_name}' with {shots_selected} shots.")
        print(f"Error: {e}")
        return None, 0, 0, 0  # Ignore this dataset and continue

    if df_partial.empty:
        return None, 0, 0, 0  # Return empty DF with 0 row counts


    # Remove unwanted 'choices_order' values
    # df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
    # # now drop the evaluation_id column
    # Compute grouped accuracy

    # processed_df = (
    #     df_unique.groupby(["dataset", "template", "separator", "enumerator", "choices_order"], as_index=False)
    #     .agg(
    #         accuracy=("score", lambda x: x.sum() / x.count()),  # Compute accuracy
    #         model_name=("model", "first"),                     # Keep first model name
    #         shots=("shots", "first"),                          # Keep first shots value
    #     )
    # )


    return df_partial, len(df_partial), len(df_partial), len(df_partial)


def immediate_error_callback(error, params):
    print("\n" + "=" * 50)
    print(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters that failed: {json.dumps(params, indent=2)}")
    print(f"Error message: {str(error)}")
    print("Traceback:")
    print(traceback.format_exc())
    print("=" * 50 + "\n")


def process_configuration_with_immediate_error(params):
    try:
        return process_configuration(params)
    except Exception as e:
        immediate_error_callback(e, params)
        return None


def run_configuration_analysis(model_name, num_processes=4, output_filepath="results.parquet") -> None:
    """
    """
    shots_to_evaluate = [0, 5]
    interesting_datasets = [
        "ai2_arc.arc_challenge",
        "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa",
    ]

    subtasks = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    pro_subtuask = [
        "history",
        "law",
        "health",
        "physics",
        "business",
        "other",
        "philosophy",
        "psychology",
        "economics",
        "math",
        "biology",
        "chemistry",
        "computer_science",
        "engineering",
    ]
    interesting_datasets.extend(["mmlu." + name for name in subtasks])
    interesting_datasets.extend(["mmlu_pro." + name for name in pro_subtuask])

    params_list = [
        (model_name, shots_selected, dataset)
        for dataset in interesting_datasets
        for shots_selected in shots_to_evaluate
    ]

    results = []
    total_raw_rows = 0
    total_processed_rows = 0
    total_unique_rows = 0

    with Pool(processes=num_processes) as pool:
        # Process in parallel
        for res, raw_rows, processed_rows, unique_rows in tqdm(
                pool.imap_unordered(process_configuration_with_immediate_error, params_list),
                total=len(params_list),
                desc="Processing configurations"
        ):
            if res is not None:
                results.append(res)
                total_raw_rows += raw_rows
                total_processed_rows += processed_rows
                total_unique_rows += unique_rows

    if not results:
        print("No results to aggregate!")
        return

    # Concatenate all processed DataFrames
    aggregated_df = pd.concat(results, ignore_index=True)
    print(f"\nAggregated results shape: {aggregated_df.shape}")

    # Save as Parquet
    aggregated_df.to_parquet(output_filepath, index=False)
    print(f"\nAggregated results saved to: {output_filepath}")

    # Print dataset size statistics
    print("\n===== Dataset Size Summary =====")
    print(f"Total raw rows (df_partial): {total_raw_rows}")
    print(f"Total processed rows (processed_df): {total_processed_rows}")
    print(f"Total unique rows (df_unique): {total_unique_rows}")
    print("================================")


if __name__ == "__main__":
    models_to_evaluate = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    for model in models_to_evaluate:
        run_configuration_analysis(model, num_processes=4, output_filepath=f"results_{model.split('/')[1]}.parquet")

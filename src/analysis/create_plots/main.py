import json
import os
from multiprocessing import Pool, Manager

from tqdm import tqdm

from src.analysis.create_plots.DataLoader import DataLoader
from src.analysis.create_plots.HammingDistanceClusterAnalyzerAxes import HammingDistanceClusterAnalyzerAxes
from src.analysis.create_plots.ModelPerformanceAnalyzer import ModelPerformanceAnalyzer
from src.analysis.create_plots.PromptConfigurationAnalyzerAxes import PromptConfigurationAnalyzerAxes
from src.analysis.create_plots.PromptQuestionAnalyzer import PromptQuestionAnalyzer


def process_configuration(params):
    """
    Process a single configuration of model and shots count
    """
    model_name, shots_selected, interesting_datasets = params
    # for Debugging:
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Processing model: {model_name} with {shots_selected} shots")

    analyzer = PromptConfigurationAnalyzerAxes()
    hamming = HammingDistanceClusterAnalyzerAxes()
    prompt_question_analyzer = PromptQuestionAnalyzer()
    performance_analyzer = ModelPerformanceAnalyzer()
    # Load data for current configuration
    data_loader = DataLoader()
    df_partial = data_loader.load_and_process_data(model_name=model_name,
                                                   shots=shots_selected,
                                                   datasets=interesting_datasets,
                                                   max_samples=None)
    # if shots_selected == 5:
    #     df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
    # base_results_dir = "../app/results_local"
    if df_partial.empty:
        return
    df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
    base_results_dir = "../app/results_local"
    os.makedirs(base_results_dir, exist_ok=True)

    performance_analyzer.generate_model_performance_comparison(
        df=df_partial,
        model_name=model_name,
        shots_selected=shots_selected,
        base_results_dir=base_results_dir
    )

    filtered_datasets = analyzer.process_and_visualize_configurations(
        df=df_partial,
        model_name=model_name,
        shots_selected=shots_selected,
        interesting_datasets=interesting_datasets,
        base_results_dir=base_results_dir
    )

    interesting_datasets = list(filtered_datasets)

    hamming.perform_clustering_for_model(
        df=df_partial,
        model_name=model_name,
        shots_selected=shots_selected,
        interesting_datasets=interesting_datasets,
        base_results_dir=base_results_dir
    )

    prompt_question_analyzer.process_and_visualize_questions(
        df=df_partial,
        model_name=model_name,
        shots_selected=shots_selected,
        interesting_datasets=interesting_datasets,
        base_results_dir=base_results_dir
    )


def run_configuration_analysis(num_processes=1) -> None:
    """
    Run the main analysis pipeline in parallel for evaluating prompt configurations
    across different models and shot counts.
    """
    # Configuration parameters
    shots_to_evaluate = [0,5]
    models_to_evaluate = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    interesting_datasets = [
        "ai2_arc.arc_challenge",
        "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa",
        "mmlu.global_facts",
        "mmlu.sociology",
        "mmlu.econometrics",
        "mmlu.high_school_geography",
    ]

    # Setup results directory

    # Create parameter combinations for parallel processing

    params_list = [
        (model_name, shots_selected, interesting_datasets)
        for shots_selected in shots_to_evaluate
        for model_name in models_to_evaluate
    ]

    with Manager() as manager:
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                    pool.imap_unordered(process_configuration_with_immediate_error, params_list),
                    total=len(params_list),
                    desc="Processing configurations"
            ):
                pass


import traceback
from datetime import datetime


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
        return {'status': 'error', 'params': params, 'error': str(e)}


if __name__ == "__main__":
    run_configuration_analysis(num_processes=4)

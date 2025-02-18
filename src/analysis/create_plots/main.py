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
    model_name, shots_selected, dataset = params
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
                                                   datasets=[dataset],
                                                   max_samples=None)
    # if shots_selected == 5:
    #     df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
    # base_results_dir = "../app/results_local"
    if df_partial.empty:
        return
    df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
    base_results_dir = "../app/results_local"
    # create global path from the base results directory withou ".."
    base_results_dir = os.path.abspath(base_results_dir)

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
        interesting_datasets=[dataset],
        base_results_dir=base_results_dir
    )

    interesting_datasets = list(filtered_datasets)

    hamming.perform_clustering_for_model(
        df=df_partial,
        model_name=model_name,
        shots_selected=shots_selected,
        dataset=dataset,
        base_results_dir=base_results_dir
    )

    prompt_question_analyzer.process_and_visualize_questions(
        df=df_partial,
        model_name=model_name,
        shots_selected=shots_selected,
        dataset=dataset,
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
        # 'meta-llama/Llama-3.2-1B-Instruct',
        # 'allenai/OLMoE-1B-7B-0924-Instruct',
        # 'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        # 'mistralai/Mistral-7B-Instruct-v0.3',
    ]
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
    interesting_datasets.extend(["mmlu."+ name for name in subtasks])
    interesting_datasets.extend(["mmlu_pro." + name for name in pro_subtuask])

    # Setup results directory

    # Create parameter combinations for parallel processing
    # model_name, shots_selected, dataset
    params_list = [
        (model_name, shots_selected, dataset)
        for dataset in interesting_datasets
        for shots_selected in shots_to_evaluate
        for model_name in models_to_evaluate
    ]

    #
    # interesting_datasets = ['hellaswag']
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # models_to_evaluate = [model_name]
    # shots_selected = 0
    # templates = [
    #     "MultipleChoiceTemplatesInstructionsStandard",
    #     "MultipleChoiceTemplatesInstructionsWithoutTopicHarness",
    #     "MultipleChoiceTemplatesInstructionsProSACould"
    # ]
    # params_list = [(model_name, shots_selected, dataset) for dataset in interesting_datasets]
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

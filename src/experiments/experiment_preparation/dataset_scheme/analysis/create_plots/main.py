import os

from src.experiments.experiment_preparation.dataset_scheme.analysis.new.DataLoader import DataLoader
from src.experiments.experiment_preparation.dataset_scheme.analysis.new.HammingDistanceClusterAnalyzerAxes import \
    HammingDistanceClusterAnalyzerAxes
from src.experiments.experiment_preparation.dataset_scheme.analysis.new.PromptConfigurationAnalyzerAxes import \
    PromptConfigurationAnalyzerAxes


def run_configuration_analysis() -> None:
    """
    Run the main analysis pipeline for evaluating prompt configurations
    across different models and shot counts.
    """
    # Configuration parameters
    shots_to_evaluate = [0, 5]
    models_to_evaluate = [
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    interesting_datasets = [
        "ai2_arc.arc_challenge",
        "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa"
    ]

    # Initialize components
    data_loader = DataLoader()
    analyzer = PromptConfigurationAnalyzerAxes()
    hamming = HammingDistanceClusterAnalyzerAxes()
    # Setup results directory
    base_results_dir = "results_local"
    os.makedirs(base_results_dir, exist_ok=True)

    # Process each combination of shots and models
    for shots_selected in shots_to_evaluate:
        print(f"\nProcessing configurations with {shots_selected} shots\n")

        for model_name in models_to_evaluate:
            print(f"Processing model: {model_name}")

            # Load data for current configuration
            df_partial = data_loader.load_and_process_data(model_name, shots_selected)

            # Analyze and visualize results
            # datasets = ['hellaswag']
            # interesting_datasets = datasets
            filtered_datasets = analyzer.process_and_visualize_configurations(
                df=df_partial,
                model_name=model_name,
                shots_selected=shots_selected,
                interesting_datasets=interesting_datasets,
                base_results_dir=base_results_dir
            )
            # Analyze and visualize results

            datasets = list(filtered_datasets)
            hamming.perform_clustering_for_model(
                df=df_partial,
                model_name=model_name,
                shots_selected=shots_selected,
                interesting_datasets=datasets,
                base_results_dir=base_results_dir
            )


if __name__ == "__main__":
    run_configuration_analysis()

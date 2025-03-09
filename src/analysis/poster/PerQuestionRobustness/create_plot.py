import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histogram(df, save_path=None):
    # Set the style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Create figure and axis with specific size
    fig = plt.figure(figsize=(13, 9))
    ax = plt.gca()
    
    # Make background transparent and remove frame
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Convert fraction to percentage
    percentage_correct = df['fraction_correct'] * 100
    bins = np.arange(0, 105, 5)  # Going to 105 to include 100 in the last bin

    # Create histogram
    n, bins, patches = plt.hist(percentage_correct,
                                bins=bins,
                                edgecolor='black',
                                color='skyblue',
                                )

    # Color the bars based on bin values
    for i in range(len(patches)):
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center >= 90:
            patches[i].set_facecolor('green')
        elif bin_center <= 10:
            patches[i].set_facecolor('red')
        else:
            patches[i].set_facecolor('skyblue')

    # Add count labels above each bar with larger, bold font
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2,
                 patches[i].get_height(),
                 f'{int(n[i])}',
                 ha='center',
                 va='bottom',
                 color='darkblue',
                 fontsize=18,
                 fontweight='bold')

    # Set axis labels
    plt.xlabel('Success Rate Over Different Prompts (%)',
               fontsize=32)
    plt.ylabel('Number of Instances in Dataset',
               fontsize=32)

    # Set x-axis range to 0-100
    plt.xlim(0, 100)

    # Increase tick label size
    plt.xticks(fontsize=24)
    plt.yticks([])  # Remove y-axis ticks and labels

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    print(f"Total samples in dataset: {len(df)}")

    # Save if path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        save_name = os.path.basename(save_path)
        if "_" in save_name:
            save_name = save_name.split('_')[1]

        savepath = os.path.join(save_dir, "hist_3" + save_name + ".png")
        plt.savefig(savepath, dpi=300, bbox_inches='tight', transparent=True)

    return plt.gcf()


def get_df(model_path):
    dfs = []
    for file in os.listdir(model_path):
        # if "_3_models" not in file:
        #     continue
        file_path = os.path.join(model_path, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[df['total_configurations'] > 3000]
    plot_histogram(df, model_path)


def main():
    """
    Run the main analysis pipeline in parallel for evaluating prompt configurations
    across different models and shot counts.
    """
    # Configuration parameters
    models_to_evaluate = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    base_path = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/app/results_local/Shots/'
    for model in models_to_evaluate:
        model_name = model.replace("/", "_")
        model_path = os.path.join(base_path, model_name)
        get_df(model_path)


def main_all_models():
    """
    Run the main analysis pipeline in parallel for evaluating prompt configurations
    across different models and shot counts.
    """
    # Configuration parameters

    base_path = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/app/results_local/Shots/Models'
    get_df(base_path)


if __name__ == "__main__":
    main()
    main_all_models()

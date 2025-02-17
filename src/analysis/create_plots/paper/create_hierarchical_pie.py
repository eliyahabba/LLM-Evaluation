import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (11, 9)  # Decreased from (15, 15)


def format_value(val):
    if val >= 1000:
        return f'{val / 1000:.1f}K'
    return str(int(val))


def create_dataset_distribution():
    # Reorganized dataset structure
    data = {
        'Logical Reasoning ': {
            'ARC': 200,
            'OpenBook_QA': 100
        },
        'STEM': {
            'MMLU-STEM': 2400,
        },
        'World Knowledge': {
            'MMLU-Other': 1700,
        },
        'Humanities': {
            'MMLU-Humanities': 1600,
        },
        'Social Sciences': {
            'MMLU-Social Sciences': 1500,
        },

        'Reading Comprehension': {
            'RACE-High': 100,
            'RACE-Middle': 100
        },
        'Commonsense': {
            'Hellaswag': 100,
            'Social_IQA': 100
        }
    }

    fig, ax = plt.subplots()

    # Calculate total sizes
    outer_sizes = {k: sum(v.values()) for k, v in data.items()}
    total = sum(outer_sizes.values())

    # Color scheme
    colors = sns.color_palette("husl", n_colors=len(data))

    # Create single ring pie chart
    wedges, texts, autotexts = ax.pie(
        outer_sizes.values(),
        labels=[f'{k}' for k, v in outer_sizes.items()],
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=0.772,
        radius=0.85,
        labeldistance=1.15,  # Add this line to push labels further out
        wedgeprops=dict(edgecolor='white', linewidth=2),  # Removed the width parameter
         textprops={'fontsize': 16}  # Increased from 12
    )
    for i, text in enumerate(texts):
        pos = text.get_position()
        if "Reading Comprehension" in text.get_text():
            text.set_position((pos[0], pos[1] - 0.15))
        elif "Logical" in text.get_text():
            text.set_position((pos[0], pos[1] + 0.35))
        else:
            text.set_position((pos[0], pos[1] + 0.05))
    # Increase the size of the percentage numbers and labels
    plt.setp(autotexts, size=20, weight="bold")
    plt.setp(texts, size=25)

    legend_labels = []
    for category, subcats in data.items():
        subcat_strings = [f"{subcat}: {format_value(size)}"
                          for subcat, size in subcats.items()]
        legend_labels.append(f"{category}:\n  " + "\n  ".join(subcat_strings))



    plt.tight_layout()
    plt.savefig('dataset_distribution_kb_split.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('dataset_distribution_kb_split.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    create_dataset_distribution()

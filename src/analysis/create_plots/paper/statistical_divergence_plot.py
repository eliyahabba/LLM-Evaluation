import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from typing import List, Dict, Tuple, Any

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
interesting_datasets = []
interesting_datasets.extend(["mmlu." + name for name in subtasks])
# interesting_datasets.extend(["mmlu_pro." + name for name in pro_subtuask])

def format_dataset_string(dataset_string):
    mapping = {
        'ai2_arc.arc_challenge': "ARC\nChallenge",
        'ai2_arc.arc_easy': "ARC\nEasy",
        'hellaswag': "HellaSwag",
        'openbook_qa': "OpenBookQA",
        'social_iqa': "Social\nIQa",
        # 'mmlu.college_biology': "MMLU\nCollege Biology",
        # 'mmlu_pro.law': "MMLU-Pro\nLaw"
    }
    if dataset_string in mapping:
        return mapping[dataset_string]
    if "." not in dataset_string:
        return dataset_string
    dataset, category = dataset_string.split('.')  # מחלק את השם ל"דאטהסט" ו"קטגוריה"

    # מחלק את שם הדאטהסט לפי "_" ומוסיף ".\n" אחרי כל חלק
    dataset_parts = dataset.split('_')
    formatted_dataset = '.\n'.join(dataset_parts) + '.\n'

    # מחלק את הקטגוריה לפי "_" ומוסיף "-\n" אחרי כל חלק
    category_parts = category.split('_')
    if     "macroeconomics" in category_parts:
        del category_parts[category_parts.index("macroeconomics")]
        category_parts.append("macro-")
        category_parts.append("economics")

    if "microeconomics" in category_parts:
        del category_parts[category_parts.index("microeconomics")]
        category_parts.append("micro-")
        category_parts.append("economics")
    if  "mathematics" in category_parts:
        del category_parts[category_parts.index("mathematics")]
        category_parts.append("math-")
        category_parts.append("ematics")
    if "international" in category_parts:
        del category_parts[category_parts.index("international")]
        category_parts.append("inter-")
        category_parts.append("national")
    if "jurisprudence" in category_parts:
        del category_parts[category_parts.index("jurisprudence")]
        category_parts.append("juris-")
        category_parts.append("prudence")
    if "professional" in category_parts:
        del category_parts[category_parts.index("professional")]
        category_parts.append("pro-")
        category_parts.append("fessional")
    if "psychology" in category_parts:
        del category_parts[category_parts.index("psychology")]
        category_parts.append("psy-")
        category_parts.append("chology")


# [Previous loading and statistical functions remain the same]
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet("aggregated_results.parquet")

    models = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ]
    df = df[df["model_name"].isin(models)]
    df = df[
        ((df['shots'] != 5) | (~df['choices_order'].isin(["correct_first", "correct_last"])))
    ]

    avilable_datasets = df['dataset'].unique()
    avilable_datasets = df['dataset'].unique()
    not_mmlu = [dataset for dataset in avilable_datasets if not dataset.startswith('mmlu')]
    mmlu = [dataset for dataset in avilable_datasets if
            dataset.startswith('mmlu') and not dataset.startswith('mmlu_pro')]
    mmlu_pro = [dataset for dataset in avilable_datasets if dataset.startswith('mmlu_pro')]
    # take random 1 mmlu pro datasets
    chosen_mmlu_pro = np.random.choice(mmlu_pro, 1)
    # take random 3 mmlu datasets
    chosen_mmlu = np.random.choice(mmlu, 5)
    chosen_datasets = [*not_mmlu, *chosen_mmlu, *chosen_mmlu_pro ,"mmlu_pro.law"]
    datasets = [
        # "hellaswag",
        # "ai2_arc.arc_challenge",
        # "ai2_arc.arc_easy",
        # "openbook_qa",
        # "social_iqa",
        # "mmlu.astronomy", "mmlu.econometrics", "mmlu.jurisprudence", "mmlu.management",
        "mmlu.prehistory",
        "mmlu.college_biology",
        "mmlu.human_aging",
        "mmlu_pro.health",
        "mmlu_pro.law",
        "mmlu_pro.psychology",
    ]
    fixed_datasets = [
        "mmlu.high_school_chemistry",
        "mmlu.high_school_statistics",
        "mmlu.international_law",
        "mmlu.moral_disputes",
        "mmlu.professional_psychology"
    ]

    datasets = [dataset for dataset in fixed_datasets]
    df = df[df['dataset'].isin(datasets)]
    # take only 0 shots
    df = df[df['shots'] == 5]
    return df

def _compute_factor_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(['model_name','dataset'])['accuracy'].agg([
        'count',
        'mean',
        'std',
        'min',
        'max',
        'median',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ]).round(2)

    stats = stats.rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'})
    return stats


def create_performance_matrix(stats_df: pd.DataFrame,origin_accuracy) -> np.ndarray:
    models = stats_df.index.get_level_values('model_name').unique()
    factor_values = stats_df.index.get_level_values(1).unique()

    performance_matrix_mean = np.zeros((len(factor_values), len(models)))
    performance_matrix_std = np.zeros((len(factor_values), len(models)))
    performance_matrix_min = np.zeros((len(factor_values), len(models)))
    performance_matrix_max = np.zeros((len(factor_values), len(models)))
    performance_matrix_median = np.zeros((len(factor_values), len(models)))
    origin_accuracy_new = np.zeros((len(factor_values), len(models)))

    for i, factor_value in enumerate(factor_values):
        for j, model in enumerate(models):
            performance_matrix_mean[i, j] = stats_df.loc[(model, factor_value), 'mean']
            performance_matrix_std[i, j] = stats_df.loc[(model, factor_value), 'std']
            performance_matrix_min[i, j] = stats_df.loc[(model, factor_value), 'min']
            performance_matrix_max[i, j] = stats_df.loc[(model, factor_value), 'max']
            performance_matrix_median[i, j] = stats_df.loc[(model, factor_value), 'median']
            origin_accuracy_new[i, j] = origin_accuracy.loc[(model, factor_value)]



    return (performance_matrix_mean , performance_matrix_std , performance_matrix_min , performance_matrix_max,
            performance_matrix_median,
            origin_accuracy_new)


def calculate_kendalls_w(performance_matrix: np.ndarray) -> float:
    rankings = np.zeros_like(performance_matrix)
    for i in range(performance_matrix.shape[0]):
        rankings[i] = stats.rankdata(performance_matrix[i])

    R = np.sum(rankings, axis=0)
    R_mean = np.mean(R)
    S = np.sum((R - R_mean) ** 2)

    m = performance_matrix.shape[0]
    n = performance_matrix.shape[1]

    W = (12 * S) / (m ** 2 * (n ** 3 - n))
    return W


def calculate_performance_divergence(performance_matrix_mean , performance_matrix_std , performance_matrix_min , performance_matrix_max,performance_matrix_median,origin_accuracy) -> np.ndarray:
    performance_matrix_mean *=100
    performance_matrix_std *=100
    performance_matrix_min *=100
    performance_matrix_max *=100
    performance_matrix_median *=100
    origin_accuracy *=100
    divergence_mean = (origin_accuracy - performance_matrix_mean ) / performance_matrix_std
    divergence_mean_from_max = (performance_matrix_max- performance_matrix_mean) / performance_matrix_std
    divergence_max_median = (performance_matrix_max - performance_matrix_median) / performance_matrix_std
    return divergence_mean


def analyze_format_impact(stats_df: pd.DataFrame, factor_values: List[str],origin_accuracy) -> Tuple[float, pd.DataFrame]:
    (performance_matrix_mean , performance_matrix_std , performance_matrix_min , performance_matrix_max,
        performance_matrix_median,
     origin_accuracy)= create_performance_matrix(stats_df,origin_accuracy)
    w = calculate_kendalls_w(performance_matrix_mean)
    divergence = calculate_performance_divergence(performance_matrix_mean , performance_matrix_std , performance_matrix_min , performance_matrix_max,performance_matrix_median,origin_accuracy)

    # Get model names from stats_df
    models = stats_df.index.get_level_values('model_name').unique()

    divergence_df = pd.DataFrame(
        divergence,
        columns=models,
        index=factor_values
    )

    return w, divergence_df

def fix_model_names(model_name):
    if 'Meta-Llama' in model_name:
        model_name = model_name.replace('Meta-Llama', 'Llama')
    return model_name


class DivergencePlotter:
    def __init__(self, figure_size=(8, 4)):
        self.figure_size = figure_size
        self.font_family = 'DejaVu Serif'

        # Define base red shades (from light to dark)
        self.red_shades = [
            '#fee0d2',  # Lightest red
            '#fcbba1',
            '#fc9272',
            '#fb6a4a',
            '#ef3b2c',
            '#cb181d',
            '#a50f15',
            '#67000d'  # Darkest red
        ]

        # Create symmetric colormap around white
        self.colors = self.red_shades[::-1] + ['#ffffff'] + self.red_shades

    def plot_divergence(self,
                        results: Dict[str, Any],
                        output_path: str = None):
        """
        Plot a single divergence heatmap.
        """
        # Calculate figure size based on data dimensions
        data = results['divergence'].T
        n_rows, n_cols = data.shape
        cell_size = 1.5  # גודל בסיסי לכל תא

        # הגדלת הגודל הכולל כדי לתת מקום לתוויות
        width = n_cols * cell_size + 2  # תוספת מרווח לתוויות צד
        height = n_rows * cell_size + 3  # תוספת מרווח לתוויות עליונות

        # Create figure with adjusted size
        fig, ax = plt.subplots(figsize=(width, height), facecolor='white')
        ax.set_facecolor('white')

        # Process data
        data.index = [self._shorten_model_name(name) for name in data.index]
        data.columns = [self.format_dataset_string(col) for col in data.columns]

        # Create custom colormap
        custom_cmap = sns.color_palette(self.colors, as_cmap=True)

        # Create heatmap with larger cells and font
        sns.heatmap(
            data,
            ax=ax,
            cmap=custom_cmap,
            center=0,
            vmin=-2,
            vmax=2,
            cbar=False,
            fmt='.2f',
            annot=True,
            annot_kws={
                'size': 24,  # הגדלת הפונט בתוך התאים
                'weight': 'bold',
                'family': self.font_family,
                'color': 'black',
                # set bold font



            },
            square=True,
            linewidths=0.5,
            linecolor='black'
        )

        # Format x-axis labels (dataset names)
        plt.setp(ax.get_xticklabels(),
                 rotation=0,
                 ha='center',
                 va='bottom',
                 fontsize=18,  # הגדלת פונט שמות הדאטה סטים
                 fontfamily=self.font_family,
                 fontweight='normal')

        # Format y-axis labels (model names)
        plt.setp(ax.get_yticklabels(),
                 rotation=0,
                 fontsize=18,  # הגדלת פונט שמות המודלים
                 fontfamily=self.font_family,
                 fontweight='normal')

        # Move x-axis labels to the top
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        # Add darker border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color('black')

        # Set equal aspect ratio to ensure square cells
        ax.set_aspect('equal', adjustable='box')

        # Adjust margins to fit all labels
        plt.tight_layout()

        # Fine-tune spacing to ensure labels are visible
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.95)

        # Save figure
        if output_path:
            plt.savefig(output_path,
                        bbox_inches='tight',  # שומר על התוויות
                        dpi=900,
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.1)  # מרווח מינימלי מסביב לתוויות

        return fig, ax

    def plot_divergence(self, results: Dict[str, Any], output_path: str = None):
        """
        מציג מפת חום ללא שימוש ב-seaborn, כאשר ניתן לשנות את גדלי הטקסטים
        - טקסט בתוך התאים (cell text)
        - תוויות ציר ה-x (דאטה סטים)
        - תוויות ציר ה-y (מודלים)
        - כותרת (אם יש)
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        # עיבוד הנתונים והתאמת התוויות
        data = results['divergence'].T
        data.index = [self._shorten_model_name(name) for name in data.index]
        data.columns = [self.format_dataset_string(col) for col in data.columns]

        matrix = data.values
        n_rows, n_cols = matrix.shape

        # === משתנים לשליטת גדלי הטקסטים ===
        cell_text_size = 18  # גודל הטקסט שבתאים (הערכים בתוך כל תא)
        x_axis_font_size = 14  # גודל הטקסט לתוויות ציר ה-x (דאטה סטים)
        y_axis_font_size = 14  # גודל הטקסט לתוויות ציר ה-y (מודלים)
        title_font_size = 24  # גודל הטקסט של הכותרת (אם בוחרים להוסיף כותרת)
        # =====================================

        # הגדרת גדלי התאים – ניתן לשנות לפי הצורך (cell_width יכול להיות שונה מ-cell_height)
        cell_width = 1.0
        cell_height = 0.5

        # חישוב גודל הפיגור (figsize)
        width = n_cols * cell_width + 2  # תוספת מרווח בצדדים
        height = n_rows * cell_height + 1.2  # תוספת מרווח מלמעלה ומלמטה

        fig, ax = plt.subplots(figsize=(width, height), facecolor='white')
        ax.set_facecolor('white')

        # יצירת colormap מותאם אישית מהרשימה self.colors
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", self.colors)

        # הצגת הנתונים – שימוש ב-aspect='auto' מאפשר התאמה של גובה ורוחב
        im = ax.imshow(matrix, cmap=custom_cmap, vmin=-2, vmax=2, aspect='auto')

        # ציור קווים שמפרידים בין התאים
        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)

        # הוספת הערכים לכל תא – שינוי גודל הטקסט נעשה כאן באמצעות cell_text_size
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=cell_text_size,
                        family=self.font_family,
                        color="black", fontweight='bold')

        # הגדרת תוויות הצירים
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))

        # שינוי גודל הטקסט של תוויות ציר ה-x (דאטה סטים)
        ax.set_xticklabels(data.columns, fontsize=x_axis_font_size, family=self.font_family, fontweight='normal')
        # שינוי גודל הטקסט של תוויות ציר ה-y (מודלים)
        ax.set_yticklabels(data.index, fontsize=y_axis_font_size, family=self.font_family, fontweight='normal')

        # העברת תוויות ציר ה-x לחלק העליון
        ax.xaxis.tick_top()
        ax.tick_params(length=0)  # הסרת ticks קטנים

        # (אופציונלי) הוספת כותרת עם גודל טקסט לפי title_font_size
        # ax.set_title("כותרת הגרף", fontsize=title_font_size, family=self.font_family)

        # התאמת המרווחים כך שהתוויות והגרף יוצגו בצורה נכונה
        plt.tight_layout()
        # plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.95)
        plt.subplots_adjust(top=0.85, bottom=0.15 )

        # שמירת הפיגור אם הוגדר נתיב קובץ
        if output_path:
            plt.savefig(output_path,
                        bbox_inches="tight",
                        dpi=900,
                        facecolor="white",
                        edgecolor="none",
                        pad_inches=0.1)

        return fig, ax
    def format_dataset_string(self, dataset_string):
        mapping = {
            'ai2_arc.arc_challenge': "ARC\nChallenge",
            'ai2_arc.arc_easy': "ARC\nEasy",
            'hellaswag': "HellaSwag",
            'openbook_qa': "OpenBook\nQA",
            'social_iqa': "Social\nIQa",
            # 'mmlu.college_biology': "MMLU\nCollege Biology",
            # 'mmlu_pro.law': "MMLU-Pro\nLaw"
        }
        if dataset_string in mapping:
            return mapping[dataset_string]
        if "." not in dataset_string:
            return dataset_string
        dataset, category = dataset_string.split('.')  # מחלק את השם ל"דאטהסט" ו"קטגוריה"

        # מחלק את שם הדאטהסט לפי "_" ומוסיף ".\n" אחרי כל חלק
        dataset_parts = dataset.split('_')
        formatted_dataset = '.\n'.join(dataset_parts) + '.\n'

        # מחלק את הקטגוריה לפי "_" ומוסיף "-\n" אחרי כל חלק
        category_parts = category.split('_')
        if "macroeconomics" in category_parts:
            del category_parts[category_parts.index("macroeconomics")]
            category_parts.append("macro")
            category_parts.append("economics")

        if "microeconomics" in category_parts:
            del category_parts[category_parts.index("microeconomics")]
            category_parts.append("micro")
            category_parts.append("economics")
        if "mathematics" in category_parts:
            del category_parts[category_parts.index("mathematics")]
            category_parts.append("math")
            category_parts.append("ematics")
        if "international" in category_parts:
            del category_parts[category_parts.index("international")]
            category_parts.append("inter")
            category_parts.append("national")
        if "jurisprudence" in category_parts:
            del category_parts[category_parts.index("jurisprudence")]
            category_parts.append("juris")
            category_parts.append("prudence")
        if "professional" in category_parts:
            del category_parts[category_parts.index("professional")]
            category_parts.append("pro")
            category_parts.append("fessional")
        if "psychology" in category_parts:
            del category_parts[category_parts.index("psychology")]
            category_parts.append("psy")
            category_parts.append("chology")
        if "prehistory" in category_parts:
            del category_parts[category_parts.index("prehistory")]
            category_parts.append("pre")
            category_parts.append("history")

        formatted_category = '-\n'.join(category_parts)
        return formatted_dataset + formatted_category

    def _shorten_model_name(self, name: str) -> str:
        """Shorten model names to match paper style"""
        name_map = {
            'allenai/OLMoE-1B-7B-0924-Instruct': 'OLMoE-1B-7B-\n0924-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama-3-8B-\nInstruct',
            'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral-7B-\nInstruct-v0.3',
            'meta-llama/Llama-3.2-1B-Instruct': 'Llama-3.2-1B-\nInstruct',
            'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3.2-3B-\nInstruct',


        }
        return name_map.get(name, name)





def get_origin_accuracy(df):
    # 'template' = "MultipleChoiceTemplatesInstructionsWithTopic",
    # 'enumerator' = "capitals",
    # 'separator' = "\n",
    # 'choices_order' = "False",
    # 'shots' = 0
    # filter the df according to the origin accuracy
    df2 = df[(df['template'] == "MultipleChoiceTemplatesInstructionsWithTopic") &
            (df['enumerator'] == "capitals") &

            (df['separator'] == "\n")&
             (df['shots'] == 5) &
            (df['choices_order'] == "none")]
            # ]
    # get the origin accuracy
    origin_accuracy = df2.groupby(['model_name', 'dataset'])['accuracy'].mean()
    return origin_accuracy

def cal_stats_kendall_w():
    df = load_and_prepare_data('your_data.csv')
    origin_accuracy = get_origin_accuracy(df)
    # factors = ['template', 'enumerator', 'separator', 'choices_order', 'shots']
    results = {}
    stats_df = _compute_factor_statistics(df)

    factor_values = stats_df.index.get_level_values(1).unique()
    w, divergence_df = analyze_format_impact(stats_df, factor_values,origin_accuracy)

    results = {
        'kendall_w': w,
        'divergence': divergence_df
    }

    print(f"\nResults for:")
    print(f"Kendall's W: {w:.3f}")

    return results


def main():
    # Calculate statistics and Kendall's W
    results = cal_stats_kendall_w()

    # Create visualization
    plotter = DivergencePlotter(figure_size=(8, 4))
    fig, ax = plotter.plot_divergence(
        results,
        output_path='divergence_plot.png'
    )
    plt.show()

if __name__ == "__main__":
    main()
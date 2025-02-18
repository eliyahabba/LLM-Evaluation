import pandas as pd
from datasets import load_dataset
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class DuplicateAnalysisResult:
    total_rows: int
    unique_questions: int  # Number of unique question groups
    duplicates_to_remove: int  # Number of rows that need to be removed to eliminate duplicates
    duplicate_percentage: float
    max_occurrences: int  # Maximum times any single question appears
    duplicate_examples: pd.DataFrame


class DatasetDuplicateAnalyzer:
    def __init__(self, df: pd.DataFrame, topic_column: str):
        """
        Initialize the analyzer with a DataFrame and specify the topic column name.

        Args:
            df: Input DataFrame
            topic_column: Name of the column containing topics/subjects
        """
        self.df = df
        self.topic_column = topic_column
        self.all_columns = df.columns.tolist()

        if topic_column:
            self.topics = df[topic_column].unique()
        else:
            self.topics = []

    def _calculate_metrics(self, duplicates_df: pd.DataFrame, group_columns: List[str]) -> DuplicateAnalysisResult:
        """
        Calculate standard metrics for duplicate analysis.
        Returns counts of unique groups and how many duplicates need to be removed.
        """
        total_rows = len(self.df)

        if len(duplicates_df) == 0:
            return DuplicateAnalysisResult(
                total_rows=total_rows,
                unique_questions=total_rows,
                duplicates_to_remove=0,
                duplicate_percentage=0.0,
                max_occurrences=1,
                duplicate_examples=duplicates_df
            )

        # Group by all relevant columns to count occurrences
        duplicate_frequencies = duplicates_df.groupby(group_columns).size()

        # Calculate metrics
        unique_duplicate_groups = len(duplicate_frequencies)
        total_duplicate_rows = sum(duplicate_frequencies)
        duplicates_to_remove = total_duplicate_rows - unique_duplicate_groups
        max_occurrences = duplicate_frequencies.max()
        duplicate_percentage = round((duplicates_to_remove / total_rows) * 100, 2)

        return DuplicateAnalysisResult(
            total_rows=total_rows,
            unique_questions=unique_duplicate_groups,
            duplicates_to_remove=duplicates_to_remove,
            duplicate_percentage=duplicate_percentage,
            max_occurrences=max_occurrences,
            duplicate_examples=duplicates_df
        )

    def analyze_topic_specific_duplicates(self, topic: str) -> DuplicateAnalysisResult:
        """Analyze duplicates within a specific topic."""
        topic_df = self.df[self.df[self.topic_column] == topic]
        duplicates = topic_df[topic_df.duplicated(subset=self.all_columns, keep=False)]
        return self._calculate_metrics(duplicates, self.all_columns)

    def analyze_cross_topic_duplicates(self) -> DuplicateAnalysisResult:
        """
        Analyze duplicates that appear across different topics.
        Finds questions that are identical except for their topic.
        Counts each unique question once per topic, even if it appears multiple times in that topic.
        """
        columns_without_topic = [col for col in self.all_columns if col != self.topic_column]

        # First, get one representative row per question per topic
        # This handles multiple occurrences within the same topic
        deduped_within_topic = self.df.drop_duplicates(
            subset=columns_without_topic + [self.topic_column],
            keep='first'
        )

        # Now find questions that appear in multiple topics
        question_duplicates = deduped_within_topic[
            deduped_within_topic.duplicated(subset=columns_without_topic, keep=False)
        ]

        # Filter to keep only groups where the topic values are different
        cross_topic_duplicates = (
            question_duplicates.groupby(columns_without_topic)
            .filter(lambda x: len(x[self.topic_column].unique()) > 1)
        )

        return self._calculate_metrics(cross_topic_duplicates, columns_without_topic)

    def analyze_all_duplicates(self) -> DuplicateAnalysisResult:
        """Analyze all duplicates in the dataset regardless of topic."""
        duplicates = self.df[self.df.duplicated(subset=self.all_columns, keep=False)]
        return self._calculate_metrics(duplicates, self.all_columns)

    @staticmethod
    def print_analysis_results(description: str, results: DuplicateAnalysisResult):
        """Print analysis results in a formatted way."""
        print(f"\n=== {description} ===")
        print(f"Total rows in dataset: {results.total_rows}")
        print(f"Total unique rows: {results.total_rows - results.duplicates_to_remove}")
        if results.duplicates_to_remove > 0:
            print(f"Number of questions that have duplicates: {results.unique_questions}")
            print(f"Number of duplicate rows to remove: {results.duplicates_to_remove}")
            print(f"Duplicate percentage: {results.duplicate_percentage}%")
            print(f"Maximum occurrences of any single question: {results.max_occurrences}")

    def analyze_per_topic(self, dataset_name: str):
        """Analyze duplicates for each topic separately."""
        print(f"\n=== Per-Topic {dataset_name} Analysis ===")
        for topic in self.topics:
            topic_results = self.analyze_topic_specific_duplicates(topic)
            if topic_results.duplicates_to_remove > 0:
                print(f"\nTopic: {topic}")
                print(f"Total rows in topic: {topic_results.total_rows}")
                print(f"Total unique rows: {topic_results.total_rows - topic_results.duplicates_to_remove}")
                print(f"Questions that have duplicates: {topic_results.unique_questions}")
                print(f"Duplicate rows to remove: {topic_results.duplicates_to_remove}")
                print(f"Maximum occurrences: {topic_results.max_occurrences}")


def analyze_mmlu():
    """Analyze duplicates in the MMLU dataset."""
    print("\nAnalyzing MMLU Dataset...")

    # Load and prepare dataset
    hf_dataset = load_dataset("cais/mmlu", "all")
    df = hf_dataset['test'].to_pandas()
    df['choices'] = df['choices'].apply(str)

    # Create analyzer
    analyzer = DatasetDuplicateAnalyzer(df, topic_column='subject')

    # Analyze overall duplicates
    overall_results = analyzer.analyze_all_duplicates()
    analyzer.print_analysis_results("Overall MMLU Analysis", overall_results)

    # Analyze cross-topic duplicates (questions same except for subject)
    cross_topic_results = analyzer.analyze_cross_topic_duplicates()
    analyzer.print_analysis_results("Cross-Topic MMLU Analysis (Same questions in different subjects)",
                                    cross_topic_results)

    # Analyze per-topic duplicates
    analyzer.analyze_per_topic("MMLU")


def analyze_mmlu_pro():
    """Analyze duplicates in the MMLU-Pro dataset."""
    print("\nAnalyzing MMLU-Pro Dataset...")

    # Load and prepare dataset
    hf_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    df = hf_dataset['test'].to_pandas()
    df['options'] = df['options'].apply(str)
    df.drop('question_id', axis=1, inplace=True)

    # Create analyzer
    analyzer = DatasetDuplicateAnalyzer(df, topic_column='category')

    # Analyze overall duplicates
    overall_results = analyzer.analyze_all_duplicates()
    analyzer.print_analysis_results("Overall MMLU-Pro Analysis", overall_results)

    # Analyze cross-topic duplicates (questions same except for category)
    cross_topic_results = analyzer.analyze_cross_topic_duplicates()
    analyzer.print_analysis_results("Cross-Topic MMLU-Pro Analysis (Same questions in different categories)",
                                    cross_topic_results)

    # Analyze per-topic duplicates
    analyzer.analyze_per_topic("MMLU-Pro")

def analyze_ai_arc():
    print("\nAnalyzing AI2-ARC Dataset...")


    # Load and prepare dataset
    hf_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    df = hf_dataset['test'].to_pandas()
    hf_dataset2 = load_dataset("allenai/ai2_arc", "ARC-Easy")
    df2 = hf_dataset2['test'].to_pandas()
    # concatenate the two datasets with new column 'dataset'
    df['dataset'] = 'ARC-Challenge'
    df2['dataset'] = 'ARC-Easy'
    df = pd.concat([df, df2])

    df['choices'] = df['choices'].apply(str)
    df.drop('id', axis=1, inplace=True)

    # Create analyzer
    analyzer = DatasetDuplicateAnalyzer(df, topic_column='dataset')

    # Analyze overall duplicates
    overall_results = analyzer.analyze_all_duplicates()
    analyzer.print_analysis_results("Overall AI2-ARC Analysis", overall_results)

    # Analyze cross-topic duplicates (questions same except for category)
    cross_topic_results = analyzer.analyze_cross_topic_duplicates()
    analyzer.print_analysis_results("Cross-Topic AI2-ARC Analysis (Same questions in different datasets)- Basic and Challenge",
                                    cross_topic_results)

    # Analyze per-topic duplicates
    analyzer.analyze_per_topic("AI2-ARC")



def analyze_social_i_qa():
    """Analyze duplicates in the MMLU-Pro dataset."""
    print("\nAnalyzing Social-I-QA Dataset...")

    # Load and prepare dataset
    hf_dataset = load_dataset("allenai/social_i_qa",trust_remote_code=True)
    df = hf_dataset['train'].to_pandas()

    # Create analyzer
    analyzer = DatasetDuplicateAnalyzer(df, topic_column=None)

    # Analyze overall duplicates
    overall_results = analyzer.analyze_all_duplicates()
    analyzer.print_analysis_results("Overall Social-I-QA Analysis", overall_results)

    # Analyze cross-topic duplicates (questions same except for category)
    # cross_topic_results = analyzer.analyze_cross_topic_duplicates()
    # analyzer.print_analysis_results("Cross-Topic Social-I-QA Analysis (Same questions in different categories)",
    #                                 cross_topic_results)

    # Analyze per-topic duplicates
    # analyzer.analyze_per_topic("MMLU-Pro")


def analyze_openbookqa():
    """Analyze duplicates in the MMLU-Pro dataset."""
    print("\nAnalyzing Social-I-QA Dataset...")

    # Load and prepare dataset
    hf_dataset = load_dataset("allenai/openbookqa", "main")
    df = hf_dataset['test'].to_pandas()
    df['choices'] = df['choices'].apply(str)
    df.drop('id', axis=1, inplace=True)
    # Create analyzer
    analyzer = DatasetDuplicateAnalyzer(df, topic_column=None)

    # Analyze overall duplicates
    overall_results = analyzer.analyze_all_duplicates()
    analyzer.print_analysis_results("Overall OpenbookQA Analysis", overall_results)

    # Analyze cross-topic duplicates (questions same except for category)
    # cross_topic_results = analyzer.analyze_cross_topic_duplicates()
    # analyzer.print_analysis_results("Cross-Topic Social-I-QA Analysis (Same questions in different categories)",
    #                                 cross_topic_results)

    # Analyze per-topic duplicates
    # analyzer.analyze_per_topic("MMLU-Pro")

def analyze_hellaswag():
    """Analyze duplicates in the MMLU-Pro dataset."""
    print("\nAnalyzing hellaswag Dataset...")
    hf_dataset = load_dataset("Rowan/hellaswag")
    df = hf_dataset['test'].to_pandas()
    df['endings'] = df['endings'].apply(str)
    df.drop('ind', axis=1, inplace=True)
    df.drop('source_id', axis=1, inplace=True)
    # Create analyzer
    analyzer = DatasetDuplicateAnalyzer(df, topic_column=None)

    # Analyze overall duplicates
    overall_results = analyzer.analyze_all_duplicates()
    analyzer.print_analysis_results("Overall hellaswag Analysis", overall_results)

    # Analyze cross-topic duplicates (questions same except for category)
    # cross_topic_results = analyzer.analyze_cross_topic_duplicates()
    # analyzer.print_analysis_results("Cross-Topic Social-I-QA Analysis (Same questions in different categories)",
    #                                 cross_topic_results)

    # Analyze per-topic duplicates
    # analyzer.analyze_per_topic("MMLU-Pro")


if __name__ == '__main__':
    analyze_hellaswag()
    analyze_openbookqa()
    analyze_ai_arc()
    analyze_social_i_qa()
    analyze_mmlu()
    analyze_mmlu_pro()
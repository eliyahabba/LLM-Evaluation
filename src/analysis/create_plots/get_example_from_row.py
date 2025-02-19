import pandas as pd
from datasets import load_dataset

class InstanceLoader:
    @staticmethod
    def load_dataset_by_name(dataset_name: str):
        """
        Map a dataset name (as in your get_all_datasets list) to the correct load_dataset call.

        The mapping is as follows:
          - For mmlu datasets (e.g., "mmlu.abstract_algebra"), use the repository "cais/mmlu"
            with the subset name (e.g., "abstract_algebra").
          - For mmlu_pro datasets (e.g., "mmlu_pro.history"), use "TIGER-Lab/MMLU-Pro" with the subset.
          - For specific base datasets, we map them manually.
          - For a "race" dataset (if needed), assume the format "race.high" or "race.middle".
        """
        if dataset_name.startswith("mmlu."):
            # e.g., "mmlu.abstract_algebra"
            subset = dataset_name.split(".", 1)[1]
            ds = load_dataset("cais/mmlu","all", split="test")
        elif dataset_name.startswith("mmlu_pro."):
            # e.g., "mmlu_pro.history"
            subset = dataset_name.split(".", 1)[1]
            ds = load_dataset("TIGER-Lab/MMLU-Pro", subset, split="test[:110]")
        elif dataset_name == "ai2_arc.arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test[:110]")
        elif dataset_name == "ai2_arc.arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test[:110]")
        elif dataset_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="test[:110]")
        elif dataset_name == "social_iqa":
            ds = load_dataset("allenai/social_i_qa", split="test[:110]", trust_remote_code=True)
        elif dataset_name == "openbook_qa":
            ds = load_dataset("allenai/openbookqa", split="test[:110]")
        elif dataset_name.startswith("race.high"):
            # e.g., "race.high" or "race.middle"
            ds = load_dataset("ehovy/race", "high", split="test[:110]")
        elif dataset_name.startswith("race.middle"):
            ds = load_dataset("ehovy/race", "middle", split="test[:110]")
        else:
            raise ValueError(f"Unknown dataset mapping for {dataset_name}")
        return ds

    @staticmethod
    def extract_answer_choices(example: dict, dataset_name: str) -> list:
        """
        Given an example and its dataset, return the list of answer choices.

        The rules are:
          - For ai2_arc (ARC) and openbook_qa: choices are in example['choices']['text']
          - For hellaswag: choices are in example['endings']
          - For mmlu: choices are in example['choices']
          - For mmlu_pro and race: choices are in example['options']
          - For social_iqa: choices are in the separate fields "answerA", "answerB", "answerC"
        """
        if dataset_name.startswith("ai2_arc") or dataset_name == "openbook_qa":
            return example['choices']['text']
        elif dataset_name == "hellaswag":
            return example['endings']
        elif dataset_name.startswith("mmlu_pro.") or dataset_name.startswith("race"):
            return example['options']
        elif dataset_name.startswith("social_iqa"):
            return [example['answerA'], example['answerB'], example['answerC']]
        elif dataset_name.startswith("mmlu."):
            return example['choices']
        else:
            raise ValueError(f"Unknown dataset mapping for answer choices extraction: {dataset_name}")

    @staticmethod
    def get_example_from_index(dataset_name: str, df: pd.DataFrame):
        """
        Given a DataFrame row with at least the columns 'dataset' and 'sample_index',
        load the dataset and return the example at the given index.
        """
        ds = InstanceLoader.load_dataset_by_name(dataset_name)
        closest_answers = []
        rows_to_drop = []
        for i, row in df.iterrows():
            sample_index = row["sample_index"]  # e.g., 3
            example = ds[sample_index]

            # If the loaded dataset is a dict of splits, we choose one:
            # if isinstance(ds, dict):
            #     # Prefer the "train" split if available, else use the first available split.
            #     split = "test" if "test" in ds else list(ds.keys())[0]['train']
            #     example = ds[split][int(sample_index)]
            # else:
            #     # Otherwise, assume it's a single dataset.
            #     example = ds[sample_index]
            answer_text = row['closest_answer'].split(". ")[1]
            answer_choices = InstanceLoader.extract_answer_choices(example, dataset_name)
            try:
                answer_index = answer_choices.index(answer_text)
                closest_answers.append(answer_index)
            except ValueError:
            #     # remove this row from the df
                rows_to_drop.append(i)
            #     raise ValueError(f"Answer '{answer_text}' not found in choices: {answer_choices}")


        df.drop(rows_to_drop, inplace=True)
        df['closest_answer_index'] = closest_answers
        assert len(closest_answers) == len(df)
        return closest_answers


if __name__ == "__main__":
    # Example usage:
    # Assuming you have a pandas DataFrame 'df' where each row has a 'dataset' and 'sample_index'
    # e.g. df.iloc[0] might be:
    #    evaluation_id    dataset              sample_index
    # 0  1b5dc69c933e396... ai2_arc.arc_challenge      3

    # filter_df_with_closest_answer_ind = InstanceLoader.get_example_from_index("ai2_arc.arc_challenge", df)

    pass

import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams


def extract_number(filename: str) -> int:
    """Extract the number from the filename pattern"""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    raise ValueError(f"No number found in filename: {filename}")


def create_file_mapping(exp_dir: str, template_dir: str) -> Dict[str, Tuple[Path, Path]]:
    """
    Create mapping between experiment_template files and template files
    Returns: Dict[template_number, (experiment_file_path, template_file_path)]
    """
    mapping = {}
    for instrcu_folder in os.listdir(exp_dir):
        # if not folder continue
        if not os.path.isdir(os.path.join(exp_dir, instrcu_folder)):
            continue
        for model_folder in os.listdir(os.path.join(exp_dir, instrcu_folder)):
            if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder)):
                continue
            for dataset_folder in os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder)):
                if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder)):
                    continue
                for shot_number in os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder)):
                    if not os.path.isdir(
                            os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number)):
                        continue
                    shots = os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number)
                    for system_folder in os.listdir(shots):
                        if not os.path.isdir(
                                os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number,
                                             system_folder)):
                            continue
                        system = os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number,
                                              system_folder)
                        exp_files = [os.path.join(system, file) for file in os.listdir(system)
                                     if file.startswith('experiment_template_') and file.endswith('.json')]

                        # Get all template files
                        template_dir2 = template_dir / Path(instrcu_folder) / Path(dataset_folder)
                        if not os.path.exists(template_dir2):
                            continue

                        # Get template files with full paths and check existence
                        template_files = [os.path.join(template_dir2, f) for f in os.listdir(template_dir2)
                                          if f.startswith('template_') and f.endswith('.json')
                                          and os.path.exists(os.path.join(template_dir2, f))]

                        # Create mapping based on numbers
                        for exp_file in exp_files:
                            try:
                                exp_num = extract_number(os.path.basename(exp_file))
                                # Find corresponding template file
                                template_file = next(
                                    (t for t in template_files
                                     if extract_number(os.path.basename(t)) == exp_num),
                                    None
                                )
                                if template_file:
                                    # Using the full experiment file path as the key
                                    mapping[str(exp_file)] = (Path(exp_file), Path(template_file))
                                else:
                                    print(f"Warning: No matching template file found for {exp_file}")
                            except ValueError as e:
                                print(f"Error processing {exp_file}: {str(e)}")
    return mapping



def rename_files(file_mapping: Dict[str, Tuple[Path, Path]], config_params: ConfigParams) -> None:
    """Rename files based on the mapping and template content"""
    for exp_file_path, (exp_path, template_path) in file_mapping.items():
        try:
            # Read template content to get new name
            with open(str(template_path), 'r', encoding='utf-8') as f:
                template_content = json.load(f)

            # Generate new name
            # take only specific keys from the template content: 'choices_separator', 'shuffle_choices', 'enumerator'
            keys = ['choices_separator', 'shuffle_choices', 'enumerator']
            template_content_subset = {k: template_content[k] for k in keys if k in template_content}


            new_name = config_params.generate_template_name(template_content_subset)
            # Create new paths
            exp_new_path = exp_path.parent / f"{new_name}.json"
            template_new_path = template_path.parent / f"{new_name}.json"

            if os.path.exists(exp_new_path) or os.path.exists(template_new_path):
                print(f"Warning: File with name {new_name}.json already exists. Skipping {exp_file_path}")
                continue

            try:
                # Rename both files
                os.rename(str(exp_path), str(exp_new_path))
                print(f"  Experiment: {exp_path} -> {new_name}.json")
            except OSError as e:
                print(f"Error renaming experiment file {exp_path}: {str(e)}")
                continue

            try:
                os.rename(str(template_path), str(template_new_path))
                print(f"  Template: {template_path} -> {new_name}.json")
            except OSError as e:
                # If template rename fails, try to revert experiment rename
                print(f"Error renaming template file {template_path}: {str(e)}")
                try:
                    os.rename(str(exp_new_path), str(exp_path))
                    print(f"  Reverted experiment file rename")
                except OSError as revert_error:
                    print(f"  Error reverting experiment file rename: {str(revert_error)}")

        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON in template file {template_path}: {str(e)}")
        except Exception as e:
            print(f"Error processing file {exp_file_path}: {str(e)}")


def update_and_rename_files(file_mapping: Dict[str, Tuple[Path, Path]], config_params: ConfigParams) -> None:
    """Rename files and update their content based on the mapping and template content"""
    for exp_file_path, (exp_path, template_path) in file_mapping.items():
        try:
            # Read template content to get new name
            with open(str(template_path), 'r', encoding='utf-8') as f:
                template_content = json.load(f)

            # Generate new name
            new_name = config_params.generate_template_name(template_content)

            # Get template directory name
            template_dir_name = template_path.parents[1].name
            new_template_name = f"{template_dir_name}/{new_name}"

            # Update experiment file content
            with open(str(exp_path), 'r', encoding='utf-8') as f:
                exp_content = json.load(f)

            # Update the template_name field
            exp_content['template_name'] = new_template_name

            # Write updated content back to experiment file
            with open(str(exp_path), 'w', encoding='utf-8') as f:
                json.dump(exp_content, f, indent=4)

            # Create new paths
            exp_new_path = exp_path.parent / f"{new_name}.json"
            template_new_path = template_path.parent / f"{new_name}.json"

            if os.path.exists(exp_new_path) or os.path.exists(template_new_path):
                print(f"Warning: File with name {new_name}.json already exists. Skipping {exp_file_path}")
                continue

            try:
                # Rename both files
                os.rename(str(exp_path), str(exp_new_path))
                print(f"  Experiment: {exp_path} -> {new_name}.json")
                print(f"  Updated template_name to: {new_template_name}")
            except OSError as e:
                print(f"Error renaming experiment file {exp_path}: {str(e)}")
                continue

            try:
                os.rename(str(template_path), str(template_new_path))
                print(f"  Template: {template_path} -> {new_name}.json")
            except OSError as e:
                # If template rename fails, try to revert experiment rename
                print(f"Error renaming template file {template_path}: {str(e)}")
                try:
                    os.rename(str(exp_new_path), str(exp_path))
                    print(f"  Reverted experiment file rename")
                except OSError as revert_error:
                    print(f"  Error reverting experiment file rename: {str(revert_error)}")

        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON in file {str(e.doc)}: {str(e)}")
        except Exception as e:
            print(f"Error processing file {exp_file_path}: {str(e)}")
def main():
    # הגדר את הנתיבים לשתי התיקיות
    experiment_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/results"
    template_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/"  # שנה לנתיב הנכון

    # יצירת אובייקט של ConfigParams
    config_params = ConfigParams()

    # יצירת מיפוי בין הקבצים
    print("Creating file mapping...")
    file_mapping = create_file_mapping(experiment_dir, template_dir)

    # הצגת המיפוי שנוצר
    print(f"\nFound {len(file_mapping)} matching file pairs")

    # שינוי שמות הקבצים
    print("\nRenaming files...")
    update_and_rename_files(file_mapping, config_params)

    print("\nFinished processing all files.")


if __name__ == "__main__":
    main()

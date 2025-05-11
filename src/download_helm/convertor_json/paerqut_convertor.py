import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from pathlib import Path
import re
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from collections import defaultdict


def setup_logger():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"json_to_parquet_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Extract info and other functions are the same as before

def extract_info_from_item(item):
    """Extract model name, dataset, and shots from a single JSON item."""
    # Same as original function
    info = {}

    # Extract model information
    if 'model' in item and 'model_info' in item['model']:
        if 'name' in item['model']['model_info']:
            info['model'] = item['model']['model_info']['name']

    # Extract dataset information
    if 'instance' in item and 'sample_identifier' in item['instance']:
        if 'dataset_name' in item['instance']['sample_identifier']:
            info['dataset'] = item['instance']['sample_identifier']['dataset_name']

        # If there's a task-specific dataset like MMLU subjects
        if 'hf_repo' in item['instance']['sample_identifier']:
            repo = item['instance']['sample_identifier']['hf_repo']
            # Extract specific subject for MMLU
            if 'mmlu' in repo.lower():
                # For MMLU, we can extract subject from the repo or from other fields
                if 'hf_index' in item['instance']['sample_identifier']:
                    # The subject might be part of the raw_input, but we don't want to extract random words
                    # Instead, rely on the dataset identification only
                    info['task'] = "default"
            else:
                # For non-MMLU datasets, just use default task
                info['task'] = "default"

    # Extract language
    if 'instance' in item and 'language' in item['instance']:
        info['language'] = item['instance']['language']
    else:
        info['language'] = 'en'  # Default to English

    # Extract shots
    if 'prompt_config' in item and 'dimensions' in item['prompt_config']:
        if 'shots' in item['prompt_config']['dimensions']:
            info['shots'] = item['prompt_config']['dimensions']['shots']

    # If dataset wasn't found, try to infer from content
    if 'dataset' not in info:
        # Check if it's MMLU
        if 'instance' in item and 'raw_input' in item['instance']:
            if 'raw_input' in item['instance'] and 'choices' in item['instance'].get('classification_fields', {}):
                # Check for MMLU-like structure
                if any('mmlu' in str(v).lower() for v in item['instance'].values()):
                    info['dataset'] = 'mmlu'
                elif 'openbook' in str(item['instance']).lower():
                    info['dataset'] = 'openbookqa'
                elif 'hellaswag' in str(item['instance']).lower():
                    info['dataset'] = 'hellaswag'
                elif 'arc' in str(item['instance']).lower():
                    info['dataset'] = 'ai2_arc'
                else:
                    # Generic dataset name if we can't determine
                    info['dataset'] = 'unknown_dataset'

    # If task wasn't found, set default
    if 'task' not in info:
        if 'dataset' in info:
            info['task'] = 'default'
        else:
            info['task'] = 'unknown_task'

    return info


def group_items_by_shots(content):
    """
    Group items by the number of shots they use.
    Returns a dictionary with shots as keys and lists of items as values.
    """
    # If it's not a list, wrap it in a list
    if not isinstance(content, list):
        content = [content]

    # Group by shots
    groups = defaultdict(list)

    for item in content:
        # Extract shots from the item
        shots = None
        if 'prompt_config' in item and 'dimensions' in item['prompt_config']:
            if 'shots' in item['prompt_config']['dimensions']:
                shots = item['prompt_config']['dimensions']['shots']

        # If shots not found, default to 0
        if shots is None:
            shots = 0

        groups[shots].append(item)

    return groups


def clean_model_name(model_name):
    """Clean model name to be suitable for directory name"""
    if not model_name:
        return "unknown_model"
    # Remove special characters, replace slashes with underscores
    return re.sub(r'[^\w\-]', '_', model_name)


def fix_special_characters(item):
    """
    Recursively fix special character representations in dictionaries and lists.
    Specifically, convert '\\n' to actual newline character '\n'
    """
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, str):
                # Fix escaped newlines in strings
                if value == '\\n':
                    item[key] = '\n'
            elif isinstance(value, (dict, list)):
                # Recursively fix nested structures
                item[key] = fix_special_characters(value)
        return item
    elif isinstance(item, list):
        return [fix_special_characters(element) for element in item]
    else:
        return item


def process_json_file(args):
    """Process a single JSON file - for parallel processing"""
    json_file, output_path = args
    logger = logging.getLogger(__name__)

    try:
        # Read JSON content
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {json_file}. Skipping.")
                return None

        # Group items by shots
        grouped_content = group_items_by_shots(content)

        results = []

        # Process each group separately
        for shots, items in grouped_content.items():
            # Fix special characters in all items
            fixed_items = [fix_special_characters(item.copy()) for item in items]

            # Specific fix for separator values
            for item in fixed_items:
                if 'prompt_config' in item and 'dimensions' in item['prompt_config'] and 'separator' in \
                        item['prompt_config']['dimensions']:
                    sep = item['prompt_config']['dimensions']['separator']
                    if sep == '\\n':
                        item['prompt_config']['dimensions']['separator'] = '\n'

            # Extract common info from the first item
            file_info = extract_info_from_item(fixed_items[0])

            if 'model' not in file_info or 'dataset' not in file_info:
                logger.warning(f"Missing model or dataset information in {json_file}. Trying to extract from filename.")
                # As fallback, try to extract from filename
                basename = os.path.basename(json_file)
                if 'model=' in basename:
                    model_part = basename.split('model=')[1].split(',')[0].split('.')[0]
                    file_info['model'] = model_part
                if 'dataset=' in basename:
                    dataset_part = basename.split('dataset=')[1].split(',')[0]
                    file_info['dataset'] = dataset_part

            if 'model' not in file_info or 'dataset' not in file_info:
                logger.warning(f"Still missing model or dataset info in {json_file}, group shots={shots}. Skipping.")
                continue

            # Make sure shots from the group is in file_info
            file_info['shots'] = shots

            # Get language from info
            language = file_info.get('language', 'en')

            # Clean model name for directory structure
            model_name = clean_model_name(file_info['model'])

            # Create directory structure
            model_dir = output_path / model_name
            lang_dir = model_dir / language
            shots_dir = lang_dir / f"{shots}_shots"
            shots_dir.mkdir(parents=True, exist_ok=True)

            # Determine output filename
            dataset = file_info['dataset']

            # Check for subject-specific datasets
            subject = None
            if dataset == 'mmlu' and 'instance' in fixed_items[0] and 'sample_identifier' in fixed_items[0]['instance']:
                # Try to extract MMLU subject from additional metadata
                if 'full_input' in fixed_items[0]['instance'].get('classification_fields', {}):
                    full_input = fixed_items[0]['instance']['classification_fields']['full_input']
                    if isinstance(full_input, str) and 'subject=' in full_input:
                        subject = full_input.split('subject=')[1].split(',')[0]

                # If subject not found, check if it's in the repo name
                if not subject and 'hf_repo' in fixed_items[0]['instance']['sample_identifier']:
                    repo = fixed_items[0]['instance']['sample_identifier']['hf_repo']
                    if 'mmlu/' in repo:
                        subject = repo.split('mmlu/')[1].split('/')[0]

            # Construct the task part of the filename
            if subject:
                task = subject
            else:
                # Don't use task in the filename to keep it clean
                task = ""

            # Create a new list to hold restructured data
            restructured_items = []

            try:
                # First, create the output file path
                if task:
                    output_file = shots_dir / f"{dataset}.{task}.parquet"
                else:
                    output_file = shots_dir / f"{dataset}.parquet"

                # Convert directly to pandas DataFrame
                df = pd.DataFrame(fixed_items)

                # Save to parquet with PyArrow engine
                df.to_parquet(str(output_file), engine='pyarrow')

                logger.info(f"Successfully converted to {output_file} (shots={shots}, items={len(fixed_items)})")
                results.append(str(output_file))

            except Exception as e:
                logger.error(f"Error saving parquet file: {e}")

                # Try alternate method using PyArrow directly
                try:
                    # Convert DataFrame to PyArrow Table
                    table = pa.Table.from_pandas(df)

                    # Write using PyArrow directly
                    pq.write_table(table, str(output_file))

                    logger.info(f"Successfully saved using direct PyArrow: {output_file}")
                    results.append(str(output_file))
                except Exception as pa_err:
                    logger.error(f"PyArrow direct save also failed: {pa_err}")

                    # Last resort: try CSV
                    try:
                        csv_file = output_file.with_suffix('.csv')
                        df.to_csv(str(csv_file), index=False)
                        logger.info(f"Saved as CSV instead: {csv_file}")
                        results.append(str(csv_file))
                    except Exception as csv_err:
                        logger.error(f"Also failed to save as CSV: {csv_err}")

        return results

    except Exception as e:
        logger.error(f"Error processing {json_file}: {e}")
        return None


def convert_json_to_parquet(input_dir, output_dir, num_workers=None):
    """
    Convert JSON files to Parquet files in the required directory structure.

    Structure:
    model_name/
    │   ├── language/
    │   │   └── shots_N/
    │   │       ├── benchmark.task.parquet
    """
    logger = setup_logger()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_path.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} JSON files to process")

    all_results = []

    # Use parallel processing if multiple workers specified
    if num_workers and num_workers > 1:
        logger.info(f"Using {num_workers} workers for parallel processing")
        args_list = [(json_file, output_path) for json_file in json_files]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_json_file, args) for args in args_list]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                result = future.result()
                if result:
                    all_results.extend(result)
    else:
        # Single-threaded processing
        for json_file in tqdm(json_files, desc="Processing files"):
            result = process_json_file((json_file, output_path))
            if result:
                all_results.extend(result)

    logger.info(f"Conversion completed. Created {len(all_results)} parquet files successfully.")
    return all_results


if __name__ == "__main__":
    input_data = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/converted_data'
    output_data = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/converted_data_paerqut'

    parser = argparse.ArgumentParser(description="Convert JSON files to Parquet in structured directories")
    parser.add_argument("--input-dir", help="Directory containing JSON files", default=input_data)
    parser.add_argument("--output-dir", help="Output directory for Parquet files", default=output_data)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: use sequential processing)")
    parser.add_argument("--test", help="Run a test to verify dictionary storage and newline handling",
                        action="store_true")

    args = parser.parse_args()

    if args.test:
        # Test script to verify dictionaries and newlines are handled correctly
        print("Running test for dictionary storage and newline handling...")

        # Create test data with dictionary containing newline
        test_data = [
            {
                'id': 1,
                'prompt_config': {
                    'dimensions': {
                        'separator': '\n',  # Actual newline character
                        'shots': 0
                    }
                }
            },
            {
                'id': 2,
                'prompt_config': {
                    'dimensions': {
                        'separator': '\\n',  # Escaped newline as string
                        'shots': 1
                    }
                }
            }
        ]

        # Fix the data (should convert '\\n' to '\n')
        fixed_data = [fix_special_characters(item.copy()) for item in test_data]

        print("Original data:")
        print(f"Item 1 separator: {repr(test_data[0]['prompt_config']['dimensions']['separator'])}")
        print(f"Item 2 separator: {repr(test_data[1]['prompt_config']['dimensions']['separator'])}")

        print("\nAfter fixing:")
        print(f"Item 1 separator: {repr(fixed_data[0]['prompt_config']['dimensions']['separator'])}")
        print(f"Item 2 separator: {repr(fixed_data[1]['prompt_config']['dimensions']['separator'])}")

        # Convert to DataFrame
        df = pd.DataFrame(fixed_data)

        # Save to temporary parquet
        temp_parquet = "test_newlines.parquet"
        df.to_parquet(temp_parquet, engine='pyarrow')

        # Read back
        df_read = pd.read_parquet(temp_parquet)

        print("\nAfter reading from parquet:")
        print(f"Item 1 separator: {repr(df_read['prompt_config'].iloc[0]['dimensions']['separator'])}")
        print(f"Item 2 separator: {repr(df_read['prompt_config'].iloc[1]['dimensions']['separator'])}")

        # Cleanup
        import os

        if os.path.exists(temp_parquet):
            os.remove(temp_parquet)

    else:
        # Run normal conversion
        convert_json_to_parquet(args.input_dir, args.output_dir, args.workers)
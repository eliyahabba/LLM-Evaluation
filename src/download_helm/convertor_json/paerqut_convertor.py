import os
import json
import pandas as pd
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


def extract_info_from_item(item):
    """
    Extract model name, dataset, and shots from a single JSON item.
    """
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
            # Extract common info from the first item
            file_info = extract_info_from_item(items[0])

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
            shots_dir = lang_dir / f"shots_{shots}"
            shots_dir.mkdir(parents=True, exist_ok=True)

            # Determine output filename
            dataset = file_info['dataset']

            # Check for subject-specific datasets
            subject = None
            if dataset == 'mmlu' and 'instance' in items[0] and 'sample_identifier' in items[0]['instance']:
                # Try to extract MMLU subject from additional metadata
                if 'full_input' in items[0]['instance'].get('classification_fields', {}):
                    full_input = items[0]['instance']['classification_fields']['full_input']
                    if isinstance(full_input, str) and 'subject=' in full_input:
                        subject = full_input.split('subject=')[1].split(',')[0]

                # If subject not found, check if it's in the repo name
                if not subject and 'hf_repo' in items[0]['instance']['sample_identifier']:
                    repo = items[0]['instance']['sample_identifier']['hf_repo']
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

            # For each item, reorganize to keep nested structure
            for item in items:
                # Create a new dictionary with top-level keys
                new_item = {}

                # For each top-level key, serialize the nested structure as JSON
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        # Store complex types as serialized JSON strings
                        new_item[key] = json.dumps(value)
                    else:
                        # Store simple types as is
                        new_item[key] = value

                restructured_items.append(new_item)

            # Create DataFrame from restructured items
            df = pd.DataFrame(restructured_items)

            # Write to Parquet
            if task:
                output_file = shots_dir / f"{dataset}.{task}.parquet"
            else:
                output_file = shots_dir / f"{dataset}.parquet"

            # Convert to parquet with pyarrow engine
            try:
                df.to_parquet(str(output_file), index=False, engine='pyarrow')
                logger.info(f"Successfully converted to {output_file} (shots={shots}, items={len(items)})")
                results.append(str(output_file))
            except Exception as e:
                logger.error(f"Error saving parquet file {output_file}: {e}")

                # Try to debug the dataframe
                logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                logger.debug(f"DataFrame shape: {df.shape}")

                # Try to save as CSV as a fallback
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


def read_parquet_and_restore_json(parquet_file):
    """
    Read a Parquet file and restore the nested JSON structure for complex fields.

    Args:
        parquet_file: Path to the Parquet file

    Returns:
        List of dictionaries with restored nested structure
    """
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    # Restore nested structure
    restored_items = []

    for _, row in df.iterrows():
        restored_item = {}

        for col, value in row.items():
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    # Try to parse JSON string back to object
                    restored_item[col] = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, keep as is
                    restored_item[col] = value
            else:
                restored_item[col] = value

        restored_items.append(restored_item)

    return restored_items


def convert_json_to_parquet(input_dir, output_dir, num_workers=None):
    """
    Convert JSON files to Parquet files in the required directory structure.

    Structure:
    model_name/
    │   ├── language/
    │   │   └── shots_N/
    │   │       ├── benchmark.task.parquet

    The nested structure of the JSON is preserved by serializing nested objects
    as JSON strings. Use read_parquet_and_restore_json() to restore the
    original nested structure when reading the Parquet files.
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
    parser = argparse.ArgumentParser(description="Convert JSON files to Parquet in structured directories")
    parser.add_argument("--input-dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for Parquet files")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: use sequential processing)")
    parser.add_argument("--read", help="Read a parquet file and restore the JSON structure")

    args = parser.parse_args()

    if args.read:
        # If --read option is used, read the parquet file and restore the JSON structure
        restored_items = read_parquet_and_restore_json(args.read)
        print(f"Read {len(restored_items)} items from {args.read}")
        print("First item sample (restored nested structure):")
        print(json.dumps(restored_items[0], indent=2))
    else:
        # Normal conversion process
        convert_json_to_parquet(args.input_dir, args.output_dir, args.workers)
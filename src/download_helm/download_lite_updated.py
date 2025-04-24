import os
import json
import logging
import pandas as pd
import requests
from typing import List, Optional, Dict
import colorama
from colorama import Fore, Style
import argparse
from tqdm import tqdm

# Initialize colorama
colorama.init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("helm_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HELM_Processor")


# Custom logger function with emojis and colors
def log_info(message, emoji="ℹ️"):
    logger.info(f"{emoji} {message}")
    print(f"{Fore.CYAN}{emoji} {message}{Style.RESET_ALL}")


def log_success(message, emoji="✅"):
    logger.info(f"{emoji} {message}")
    print(f"{Fore.GREEN}{emoji} {message}{Style.RESET_ALL}")


def log_error(message, emoji="❌"):
    logger.error(f"{emoji} {message}")
    print(f"{Fore.RED}{emoji} {message}{Style.RESET_ALL}")


def log_warning(message, emoji="⚠️"):
    logger.warning(f"{emoji} {message}")
    print(f"{Fore.YELLOW}{emoji} {message}{Style.RESET_ALL}")


def log_step(step_name, emoji="🔄"):
    logger.info(f"{emoji} {step_name}")
    print(f"{Fore.MAGENTA}{emoji} {step_name}{Style.RESET_ALL}")


# Configure base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "converted_data")

# Create necessary directories
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_json_from_url(url):
    """Fetch JSON data from given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # log_error(f"An error occurred with URL {url}: {e}")
        return None


def read_tasks_from_csv(csv_file: str) -> List[str]:
    """
    Read tasks from CSV file in the format:
    ,Run,Model,Groups,Adapter method,Subject / Task

    Returns a list of Run values (HELM catalog lines)
    """
    try:
        df = pd.read_csv(csv_file)
        if 'Run' not in df.columns:
            raise ValueError(f"CSV file {csv_file} doesn't have a 'Run' column")

        tasks_list = list(df['Run'])
        log_info(f"Read {len(tasks_list)} tasks from {csv_file}")
        return [task for task in tasks_list if task and isinstance(task, str)]
    except Exception as e:
        log_error(f"Error reading CSV file {csv_file}: {str(e)}")
        return []


def download_single_task(task: str, start_version: str = "v1.0.0", output_dir: str = DOWNLOADS_DIR,
                         overwrite: bool = False) -> Dict:
    """
    Download a single HELM task from multiple possible versions
    Returns statistics about what was downloaded
    """
    log_step(f"Downloading task: {task}", "🔽")

    # List of versions to check, in order
    versions = [f"v1.{i}.0" for i in range(14)]  # v1.0.0 to v1.13.0
    # Start from the specified version
    start_idx = versions.index(start_version)
    versions = versions[start_idx:]

    # Base URL template
    base_url_template = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/{version}"

    # Different file types to download for each task
    file_types = [
        "run_spec",
        "stats",
        "per_instance_stats",
        "instances",
        "scenario_state",
        "display_predictions",
        "display_requests",
        "scenario"
    ]

    # Create directory for this task with proper permissions for later deletion
    save_dir = os.path.join(output_dir, task)
    
    # Remove directory if it exists to ensure clean state
    if os.path.exists(save_dir) and overwrite:
        try:
            import shutil
            shutil.rmtree(save_dir)
            log_info(f"Removed existing directory: {save_dir}", "🗑️")
        except Exception as e:
            log_warning(f"Could not remove existing directory: {e}", "⚠️")
    
    # Create the directory with full permissions
    os.makedirs(save_dir, exist_ok=True)
    
    # Set directory permissions to 0o777 (rwxrwxrwx) to ensure it can be deleted later
    try:
        os.chmod(save_dir, 0o777)
        log_info(f"Set directory permissions to allow deletion", "🔑")
    except Exception as e:
        log_warning(f"Could not set directory permissions: {e}", "⚠️")

    # Initialize statistics for this task
    task_stats = {
        "task": task,
        "total_files": len(file_types),
        "found_files": 0,
        "missing_files": 0,
        "version_usage": {v: 0 for v in versions}
    }

    for file_type in file_types:
        save_path = os.path.join(save_dir, f"{file_type}.json")
        found = False

        # Skip if file exists and not overwriting
        if os.path.exists(save_path) and not overwrite:
            log_info(f"File {file_type}.json already exists for task {task} (skipping)", "📄")
            task_stats["found_files"] += 1
            continue

        # Try each version in order until we find the file
        for version in versions:
            cur_url = f"{base_url_template.format(version=version)}/{task}/{file_type}.json"
            json_data = get_json_from_url(cur_url)

            if json_data:
                with open(save_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                task_stats["version_usage"][version] += 1
                found = True
                task_stats["found_files"] += 1
                log_success(f"Found {task}/{file_type}.json in version {version}", "📥")
                break

        if not found:
            task_stats["missing_files"] += 1
            log_warning(f"Could not find {task}/{file_type}.json in any version", "🔍")

    # Log success/failure info
    if task_stats["found_files"] == task_stats["total_files"]:
        log_success(f"Downloaded all {task_stats['total_files']} files for task {task}", "🎉")
    else:
        log_warning(
            f"Downloaded {task_stats['found_files']}/{task_stats['total_files']} files for task {task}",
            "⚠️"
        )

    return task_stats


def download_tasks(tasks: List[str], start_version: str = "v1.0.0",
                   output_dir: str = DOWNLOADS_DIR, overwrite: bool = False) -> Dict:
    """
    Download multiple HELM tasks
    Returns statistics about what was downloaded
    """
    log_step(f"Downloading {len(tasks)} tasks", "🔽")

    # Initialize statistics
    download_stats = {
        "total_tasks": len(tasks),
        "total_files_checked": len(tasks) * 8,  # 8 file types per task
        "files_found": 0,
        "files_missing": 0,
        "version_usage": {f"v1.{i}.0": 0 for i in range(14)}
    }

    # Process each task with a progress bar
    for task in tqdm(tasks, desc="Processing tasks"):
        task_stats = download_single_task(task, start_version, output_dir, overwrite)

        # Update global statistics
        download_stats["files_found"] += task_stats["found_files"]
        download_stats["files_missing"] += task_stats["missing_files"]

        # Update version usage
        for version, count in task_stats["version_usage"].items():
            if version in download_stats["version_usage"]:
                download_stats["version_usage"][version] += count

    # Print final statistics
    log_step("Download Statistics", "📊")
    log_info(f"Total tasks processed: {download_stats['total_tasks']}")
    log_info(f"Total files checked: {download_stats['total_files_checked']}")
    log_info(f"Files found: {download_stats['files_found']}")
    log_info(f"Files missing: {download_stats['files_missing']}")

    log_step("Files found per version:", "📈")
    for version, count in download_stats["version_usage"].items():
        if count > 0:
            log_info(f"{version}: {count} files")

    return download_stats


def download_from_csv(csv_file: str, start_version: str = "v1.0.0",
                      output_dir: str = DOWNLOADS_DIR, overwrite: bool = False) -> Dict:
    """
    Download HELM tasks from a CSV file
    Returns statistics about what was downloaded
    """
    log_step(f"Downloading tasks from CSV: {csv_file}", "📋")
    tasks = read_tasks_from_csv(csv_file)
    if not tasks:
        log_error(f"No tasks found in CSV file {csv_file}")
        return {}

    return download_tasks(tasks, start_version, output_dir, overwrite)


def main(input_lines: List[str] = None, csv_file: str = None, start_version: str = "v1.0.0",
         output_dir: str = DOWNLOADS_DIR, overwrite: bool = False) -> Dict:
    """
    Main function to download HELM data
    Returns statistics about what was downloaded
    """
    if csv_file:
        return download_from_csv(csv_file, start_version, output_dir, overwrite)
    elif input_lines:
        return download_tasks(input_lines, start_version, output_dir, overwrite)
    else:
        log_error("Either csv_file or input_lines must be provided")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HELM data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv-file", help="CSV file with 'Run' column containing HELM catalog lines")
    group.add_argument("--input-line", help="Single line to process")
    parser.add_argument("--output-dir", default=DOWNLOADS_DIR, help="Directory to save downloaded files")
    parser.add_argument("--start-version", default="v1.0.0", help="Starting version to check (e.g., v1.0.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    log_info(f"Starting HELM Download Lite", "🚀")
    log_info(f"Arguments: {args}", "🔧")

    if args.csv_file:
        log_step(f"Processing from CSV file: {args.csv_file}", "📋")
        stats = main(
            csv_file=args.csv_file,
            start_version=args.start_version,
            output_dir=args.output_dir,
            overwrite=args.overwrite
        )
    elif args.input_line:
        log_step(f"Processing single line: {args.input_line}", "📝")
        stats = main(
            input_lines=[args.input_line],
            start_version=args.start_version,
            output_dir=args.output_dir,
            overwrite=args.overwrite
        )

    log_success(f"HELM Download Lite completed", "🏁")
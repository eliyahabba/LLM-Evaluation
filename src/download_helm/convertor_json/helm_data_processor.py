import os
import zipfile
import shutil
import json
import logging
import pandas as pd
from typing import List, Optional
import colorama
from colorama import Fore, Style
import sys
from tqdm import tqdm

# Add the current directory to the path if it's not already there
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Import functions from download_lite_updated.py
from download_lite_updated import (
    download_single_task, download_tasks, download_from_csv,
    log_info, log_success, log_error, log_warning, log_step
)

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

# Configure base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
EXTRACTED_DIR = os.path.join(BASE_DIR, "extracted")
OUTPUT_DIR = os.path.join(BASE_DIR, "converted_data")

# Create necessary directories
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_tasks_from_csv(csv_file: str, adapter_method: str = None) -> List[str]:
    """
    Read tasks from CSV file in the format:
    ,Run,Model,Groups,Adapter method,Subject / Task

    Parameters:
        csv_file: Path to the CSV file
        adapter_method: Optional filter value for the 'Adapter method' column
                       (e.g., 'multiple_choice_joint')

    Returns a list of Run values (HELM catalog lines)
    """
    try:
        df = pd.read_csv(csv_file)
        total_rows = len(df)
        log_info(f"CSV file contains {total_rows} total rows", "📊")
        
        if 'Run' not in df.columns:
            raise ValueError(f"CSV file {csv_file} doesn't have a 'Run' column")

        # Apply filter if adapter_method is provided
        if adapter_method and 'Adapter method' in df.columns:
            log_info(f"Filtering tasks by Adapter method: {adapter_method}", "🔍")
            filtered_df = df[df['Adapter method'] == adapter_method]
            filtered_rows = len(filtered_df)
            
            # Log the filtering results
            log_info(f"Found {filtered_rows} tasks with Adapter method '{adapter_method}' out of {total_rows} total tasks", "📋")
            log_info(f"Using {filtered_rows}/{total_rows} rows ({filtered_rows/total_rows*100:.1f}%)", "📊")
            
            if filtered_rows == 0:
                log_warning(f"No tasks found with Adapter method '{adapter_method}'", "⚠️")
                # Show available adapter methods for debugging
                available_methods = df['Adapter method'].unique()
                log_info(f"Available Adapter methods: {', '.join(str(m) for m in available_methods)}", "ℹ️")
            
            df = filtered_df

        tasks_list = list(df['Run'])
        valid_tasks = [task for task in tasks_list if task and isinstance(task, str)]
        
        log_info(f"Read {len(valid_tasks)} valid tasks from {csv_file}", "📄")
        if len(valid_tasks) < len(tasks_list):
            log_warning(f"Filtered out {len(tasks_list) - len(valid_tasks)} invalid or empty tasks", "⚠️")
            
        return valid_tasks
    except Exception as e:
        log_error(f"Error reading CSV file {csv_file}: {str(e)}", "❌")
        return []


def download_helm_data(input_lines: List[str] = None, input_file: Optional[str] = None,
                       csv_file: Optional[str] = None) -> List[str]:
    """
    Download HELM data based on provided lines, from a text file, or from a CSV file
    Returns a list of downloaded zip file paths
    """
    if csv_file and os.path.exists(csv_file):
        log_step(f"Reading tasks from CSV file: {csv_file}", "📋")
        return download_from_csv(csv_file, DOWNLOADS_DIR)
    elif input_file and os.path.exists(input_file):
        log_step(f"Reading tasks from input file: {input_file}", "📄")
        with open(input_file, 'r') as f:
            lines = f.read().splitlines()
        return download_tasks(lines, DOWNLOADS_DIR)
    elif input_lines:
        log_step(f"Processing provided input lines", "📝")
        return download_tasks(input_lines, DOWNLOADS_DIR)
    else:
        log_error("No input provided for download")
        return []


def extract_zip_file(zip_path: str) -> str:
    """
    Extract a zip file to the extraction directory
    Returns the path to the extracted directory
    """
    # Get a unique name for the extraction folder based on the zip filename
    zip_basename = os.path.basename(zip_path).replace('.zip', '')
    extraction_path = os.path.join(EXTRACTED_DIR, zip_basename)

    log_step(f"Extracting {zip_path} to {extraction_path}", "📂")

    # Create a fresh extraction directory
    if os.path.exists(extraction_path):
        log_info(f"Removing existing extraction directory", "🗑️")
        shutil.rmtree(extraction_path)
    os.makedirs(extraction_path)

    # Extract the zip file
    log_info(f"Unzipping file", "🔓")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    log_success(f"Extraction completed")

    # Find the actual data directory (which might be a subdirectory)
    log_info(f"Looking for data directory with run_spec.json and instances.json", "🔍")
    for root, dirs, files in os.walk(extraction_path):
        if 'run_spec.json' in files and 'instances.json' in files:
            log_success(f"Found data directory: {root}", "🎯")
            return root

    log_warning(f"Could not find expected data structure, using top directory", "⚠️")
    return extraction_path  # Return the top directory if we can't find the expected structure


def convert_data(data_dir: str) -> str:
    """
    Convert the extracted data using convert_cluade.py
    Returns the path to the converted JSON file
    """
    # Create a unique output filename based on the data directory
    dir_basename = os.path.basename(data_dir)
    output_file = os.path.join(OUTPUT_DIR, f"{dir_basename}_converted.json")

    log_step(f"Converting data from {data_dir} to {output_file}", "🔄")

    try:
        # Import the convert_cluade module directly
        try:
            log_info(f"Importing convert_cluade module", "📥")
            # Add the convertor_json directory to the path
            convertor_dir = os.path.join(BASE_DIR, "convertor_json")
            if convertor_dir not in sys.path:
                sys.path.append(convertor_dir)
                
            from convert_cluade import main as convert_main

            # Call the main function directly
            log_info(f"Running conversion", "⚙️")
            convert_main(data_dir, output_file)

            log_success(f"Conversion completed successfully")
            return output_file

        except ImportError as e:
            log_error(f"Error importing convert_cluade: {e}")
            return None

    except Exception as e:
        log_error(f"Conversion failed: {e}")
        return None


def cleanup(zip_path: str) -> None:
    """
    Clean up temporary files and directories
    """
    log_step(f"Cleaning up temporary files", "🧹")

    # Remove zip file
    if os.path.exists(zip_path):
        shutil.rmtree(zip_path) if os.path.isdir(zip_path) else os.remove(zip_path)
        log_info(f"Removed ZIP file: {zip_path}", "🗑️")


def process_line(line: str, keep_temp_files: bool = False, overwrite: bool = False) -> dict:
    """
    Process a single line from start to finish
    Returns a dictionary with the results
    """
    log_step(f"Processing line: {line}", "🔄")
    
    result = {
        "input_line": line,
        "status": "skipped",  # Default to skipped
        "downloaded_file": None,
        "extracted_dir": None,
        "converted_file": None,
        "entry_count": 0,
        "error": None
    }

    # Create a predictable output filename based on the line
    expected_output_file = os.path.join(OUTPUT_DIR, f"{line}_converted.json")
    
    # Check if the output file already exists
    if os.path.exists(expected_output_file) and not overwrite:
        log_info(f"Output file already exists: {expected_output_file}", "📋")
        
        # Count entries in the existing file
        try:
            with open(expected_output_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                entry_count = count_json_entries(json_data)
                result["entry_count"] = entry_count
                log_info(f"File contains {entry_count} entries", "🔢")
        except Exception as e:
            log_warning(f"Could not count entries in existing file: {e}", "⚠️")
        
        log_success(f"Skipping processing for line: {line} (output file already exists)", "⏭️")
        result["converted_file"] = expected_output_file
        return result

    # If we get here, we need to process the line
    result["status"] = "failed"  # Reset status to failed for processing

    try:
        # Download the data
        log_step(f"Starting download phase", "🔽")
        downloaded_files = download_tasks([line], output_dir=DOWNLOADS_DIR, overwrite=overwrite)
        if not downloaded_files:
            log_error(f"No files were downloaded")
            raise ValueError("No files were downloaded")

        saved_dir = os.path.join(DOWNLOADS_DIR, line)
        result["extracted_dir"] = saved_dir

        # Convert the data
        log_step(f"Starting conversion phase", "🔄")
        converted_file = convert_data(saved_dir)
        if not converted_file:
            log_error(f"Conversion failed")
            raise ValueError("Conversion failed")

        result["converted_file"] = converted_file
        
        # Count entries in the converted file
        try:
            with open(converted_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                entry_count = count_json_entries(json_data)
                result["entry_count"] = entry_count
                log_info(f"Converted file contains {entry_count} entries", "🔢")
        except Exception as e:
            log_warning(f"Could not count entries in converted file: {e}", "⚠️")
            
        log_success(f"Conversion phase completed: {converted_file}", "📄")

        # Clean up if not keeping temp files
        if not keep_temp_files:
            log_step(f"Starting cleanup phase", "🧹")
            cleanup(saved_dir)
            log_success(f"Cleanup phase completed", "🧹")
        else:
            log_info(f"Skipping cleanup (keep_temp_files=True)", "🚫")

        result["status"] = "success"
        log_success(f"Processing completed successfully for line: {line}", "🎉")

    except Exception as e:
        log_error(f"Error processing line '{line}': {str(e)}")
        result["error"] = str(e)

    return result


def count_json_entries(json_data):
    """
    Count the number of entries in a JSON object
    Handles different JSON structures that might be encountered
    """
    # If it's a list, return its length
    if isinstance(json_data, list):
        return len(json_data)
    
    # If it's a dict with an 'instances' key that is a list, return its length
    if isinstance(json_data, dict) and 'instances' in json_data and isinstance(json_data['instances'], list):
        return len(json_data['instances'])
    
    # If it's a dict with a 'data' key that is a list, return its length
    if isinstance(json_data, dict) and 'data' in json_data and isinstance(json_data['data'], list):
        return len(json_data['data'])
    
    # If it's a dict, count all values that are lists
    if isinstance(json_data, dict):
        list_counts = [len(v) for k, v in json_data.items() if isinstance(v, list)]
        if list_counts:
            return max(list_counts)  # Return the largest list count
    
    # If we can't determine a count, return 0
    return 0


def main(input_file: str = None, input_lines: List[str] = None, csv_file: str = None, 
         adapter_method: str = None, keep_temp_files: bool = False, overwrite: bool = False):
    """
    Main function to process all lines
    """
    # Process input from CSV, file, or list
    if csv_file:
        lines = read_tasks_from_csv(csv_file, adapter_method)
    elif input_file:
        with open(input_file, 'r') as f:
            lines = f.read().splitlines()
    elif input_lines:
        lines = input_lines
    else:
        raise ValueError("Either csv_file, input_file, or input_lines must be provided")

    # Filter out empty lines
    lines = [line for line in lines if line and isinstance(line, str) and line.strip()]
    
    if not lines:
        log_warning("No tasks to process after filtering", "⚠️")
        return []

    results = []
    total_entries = 0
    
    # Create a colorful progress bar
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    # Configure colors for the progress bar
    bar_format = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    
    # Use tqdm with logging redirect to prevent progress bar disruption by log messages
    with logging_redirect_tqdm():
        # Create the progress bar
        pbar = tqdm(
            total=len(lines),
            desc=f"Processing {len(lines)} tasks",
            bar_format=bar_format,
            colour='green',
            ncols=100
        )
        
        for i, line in enumerate(lines):
            # Update progress bar description with current task
            pbar.set_description(f"Task {i+1}/{len(lines)}: {line[:30]}{'...' if len(line) > 30 else ''}")
            
            log_info(f"Processing line {i + 1}/{len(lines)}: {line}")
            result = process_line(line, keep_temp_files=keep_temp_files, overwrite=overwrite)
            results.append(result)
            total_entries += result.get("entry_count", 0)

            # Log the result
            if result["status"] == "success":
                status_emoji = "✅"
                pbar.colour = 'green'  # Success is green
            elif result["status"] == "skipped":
                status_emoji = "⏭️"
                pbar.colour = 'blue'   # Skipped is blue
            else:
                status_emoji = "❌"
                pbar.colour = 'red'    # Failed is red
                
            # Update progress bar with status and entry count
            pbar.set_postfix(
                status=result["status"], 
                entries=result.get("entry_count", 0),
                total_entries=total_entries
            )
            
            log_info(f"{status_emoji} Completed processing line {i + 1}: {line} ({result.get('entry_count', 0)} entries)")
            
            # Update the progress bar
            pbar.update(1)
        
        # Close the progress bar when done
        pbar.close()

    # Generate a summary report
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    
    log_info(f"Processed {len(results)} lines:")
    log_info(f"  ✅ Successful: {success_count}")
    log_info(f"  ⏭️ Skipped: {skipped_count}")
    log_info(f"  ❌ Failed: {failed_count}")
    log_info(f"  🔢 Total entries across all files: {total_entries}")

    # Save results to a JSON file
    results_file = os.path.join(OUTPUT_DIR, "processing_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_lines": len(results),
                "successful": success_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "total_entries": total_entries
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    log_info(f"Results saved to {results_file}")
    log_success(f"Total entries processed: {total_entries}", "🔢")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process HELM data from download to conversion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv-file", help="CSV file with 'Run' column containing HELM catalog lines")
    group.add_argument("--input-file", help="Text file with lines to process, one per line")
    group.add_argument("--input-line", help="Single line to process")
    parser.add_argument("--adapter-method", help="Filter tasks by Adapter method (e.g., 'multiple_choice_joint')")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files (zip and extracted directories)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    args = parser.parse_args()
    
    log_info(f"Starting HELM Data Processor", "🚀")
    log_info(f"Arguments: {args}", "🔧")

    if args.csv_file:
        log_step(f"Processing from CSV file: {args.csv_file}", "📋")
        main(csv_file=args.csv_file, adapter_method=args.adapter_method, 
             keep_temp_files=args.keep_temp, overwrite=args.overwrite)
    elif args.input_file:
        log_step(f"Processing from input file: {args.input_file}", "📄")
        main(input_file=args.input_file, keep_temp_files=args.keep_temp, overwrite=args.overwrite)
    elif args.input_line:
        log_step(f"Processing single line: {args.input_line}", "📝")
        main(input_lines=[args.input_line], keep_temp_files=args.keep_temp, overwrite=args.overwrite)
        
    log_success(f"HELM Data Processor completed", "🏁")
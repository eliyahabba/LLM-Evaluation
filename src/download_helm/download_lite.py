from tqdm import tqdm
import os
import requests
import json
import pandas as pd


def get_json_from_url(url):
    """Fetch JSON data from given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # print(f"An error occurred: {e}")
        return None


def download_helm_lite(start_version="v1.0.0", save_base_dir="helm/lite", overwrite=False):
    """Download HELM Lite benchmark data, checking multiple versions if data not found."""

    # Read tasks list from CSV
    df = pd.read_csv('helm_data.csv')
    tasks_list = list(df.Run)

    # List of versions to check, in order
    versions = [f"v1.{i}.0" for i in range(14)]  # v1.0.0 to v1.13.0
    # Start from the specified version
    start_idx = versions.index(start_version)
    versions = versions[start_idx:]

    # Base URL template
    base_url_template = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/{version}"

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

    # Initialize counters for statistics
    total_files = len(tasks_list) * len(file_types)
    found_files = 0
    missing_files = 0
    version_usage = {v: 0 for v in versions}

    for task in tqdm(tasks_list, desc="Processing tasks"):
        for file_type in file_types:
            found = False

            # Create directory for this task
            save_dir = f"/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/downloads"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{file_type}.json"

            # Skip if file exists and not overwriting
            if os.path.exists(save_path) and not overwrite:
                found_files += 1
                continue

            # Try each version in order until we find the file
            for version in versions:
                cur_url = f"{base_url_template.format(version=version)}/{task}/{file_type}.json"
                json_data = get_json_from_url(cur_url)

                if json_data:
                    json.dump(json_data, open(save_path, "w"), indent=2)
                    version_usage[version] += 1
                    found = True
                    found_files += 1
                    print(f"Found {task}/{file_type}.json in version {version}")
                    break

            if not found:
                missing_files += 1
                print(f"Warning: Could not find {task}/{file_type}.json in any version")

    # Print statistics
    print("\n--- Download Statistics ---")
    print(f"Total files checked: {total_files}")
    print(f"Files found: {found_files}")
    print(f"Files missing: {missing_files}")
    print("\nFiles found per version:")
    for version, count in version_usage.items():
        if count > 0:
            print(f"{version}: {count} files")


if __name__ == "__main__":
    download_helm_lite()
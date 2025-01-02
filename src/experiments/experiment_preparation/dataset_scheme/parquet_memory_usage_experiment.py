import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def restructure_prompt_logprobs(row):
    data = eval(row) if isinstance(row, str) else row

    if data[0] == 'None':
        data = data[1:]

    arrays = {
        'token_id': [],
        'logprob': [],
        'rank': [],
        'decoded_token': []
    }

    for pos, item_dict in enumerate(data):
        if item_dict == 'None' or item_dict is None:
            continue
        logprob = []
        tokens_id = []
        rank = []
        decoded_token = []
        for token_id, token_data in item_dict.items():
            logprob.append(token_data['logprob'])
            tokens_id.append(int(token_id))
            rank.append(token_data['rank'])
            decoded_token.append(token_data['decoded_token'])
        arrays['token_id'].append(tokens_id)
        arrays['rank'].append(rank)
        arrays['decoded_token'].append(decoded_token)
        arrays['logprob'].append(logprob)

    return {
        'token_id': arrays['token_id'],
        'logprob': arrays['logprob'],
        'rank': arrays['rank'],
        'decoded_token': arrays['decoded_token']
    }


def restructure_scores(scores):
    # Convert the string to a Python object (e.g., list, dict)
    scores = eval(scores) if isinstance(scores, str) else scores
    scores_dict = {
        'accuracy': scores.get('accuracy'),
        'score': scores.get('score'),
        'score_name': scores.get('score_name')
    }
    scores_dict['accuracy'] = int(scores_dict['accuracy'])
    scores_dict['score'] = int(scores_dict['score'])
    return scores_dict


def _process_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Process the scores column into separate columns."""
    restructured = df['scores'].apply(restructure_scores)
    df[['accuracy', 'score', 'score_name']] = pd.DataFrame(
        restructured.tolist(),
        index=df.index
    )
    return df.drop(columns=['scores'])


def _process_logprobs_column(df: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
    """
    Process logprobs or prompt_logprobs columns.

    Args:
        df: Input DataFrame
        column_name: Name of column to process ('logprobs' or 'prompt_logprobs')
        prefix: Prefix for new column names
    """
    processed_series = df[column_name].apply(restructure_prompt_logprobs)

    # Create new columns for each key in the processed data
    for key in processed_series.iloc[0].keys():
        new_column = f'{prefix}_{key}'
        df[new_column] = processed_series.apply(lambda x: x[key])

    return df.drop(columns=[column_name])


def _convert_decoded_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all decoded token columns to string type."""
    decoded_cols = [col for col in df.columns if "decoded" in col]
    for col in decoded_cols:
        df[col] = df[col].astype(str)
    return df


def _process_token_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Process token ID columns from string to evaluated form."""
    token_id_columns = [
        'generated_text_tokens_ids',
        'prompt_token_ids'
    ]

    for column in token_id_columns:
        if column in df.columns:
            df[column] = df[column].apply(eval)

        return df


def _process_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process logprobs or prompt_logprobs columns.

    Args:
        df: Input DataFrame
        column_name: Name of column to process ('logprobs' or 'prompt_logprobs')
        prefix: Prefix for new column names
    """
    processed_series = df['raw_input'].apply(_restructure_raw_input)
    df['raw_input'] = processed_series
    df.drop(columns=['raw_input'])
    # Create new columns for each key in the processed data
    return df
# dict_keys(['source', 'target', 'references', 'metrics', '', 'subset', 'media', 'postprocessors', 'task_data', 'data_classification_policy'])
# 'source', 'target', 'references', metrics,groups,  '', 'subset', 'media','data_classification_policy


def _restructure_raw_input(row):
    data = eval(row) if isinstance(row, str) else row
# delete these keys from the dictionary
    #'source', 'target', 'references', metrics, groups, '', 'subset', 'media', 'data_classification_policy
    keys_to_delete = ['source', 'target', 'references', 'metrics', 'groups', 'subset', 'media', 'data_classification_policy']
    for key in keys_to_delete:
        if key in data:
            del data[key]
    return data


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a DataFrame containing model output data.

    Transforms various columns including scores, logprobs, and token IDs
    into a more structured format.

    Args:
        df: Input DataFrame with raw model output

    Returns:
        pd.DataFrame: Processed DataFrame with restructured columns
    """
    # Process each component
    df = _process_scores(df)

    df = _process_logprobs_column(
        df,
        column_name='logprobs',
        prefix='logprobs',

    )

    df = _process_logprobs_column(
        df,
        column_name='prompt_logprobs',
        prefix='prompt_logprobs'
    )

    df = _convert_decoded_columns_to_string(df)
    df = _process_token_ids(df)
    df = _process_raw_input(df)

    return df


def get_batch_df(input_path, batch_size: int) -> Iterator[pd.DataFrame]:
    parquet_file = pq.ParquetFile(input_path)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        yield df


def process_parquet_file(
        input_path,
        output_path,
        batch_limit: int,
        batch_size: int,
        process_func=None
) -> None:
    """
    Process a parquet file in batches and save the result.

    Args:
        input_path: Path to input parquet file
        output_path: Path to save processed data
        batch_limit: Maximum number of batches to process
        process_func: Optional function to process each batch
    """

    processed_dfs = []

    for batch_num, batch in enumerate(get_batch_df(input_path=input_path, batch_size=batch_size)):
        print(f"Processing batch {batch_num}, shape: {batch.shape}")

        if process_func:
            batch = process_func(batch)

        processed_dfs.append(batch)

        if batch_num >= batch_limit - 1:
            break

    # Combine and save results
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    table = pa.Table.from_pandas(combined_df)
    pq.write_table(table, Path(output_path))


def compare_file_sizes(file1, file2) -> None:
    """
    Compare and print sizes of two files.

    Args:
        file1: Path to first file
        file2: Path to second file
    """
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)

    print(f"Size of {file1}: {size1:,} bytes")
    print(f"Size of {file2}: {size2:,} bytes")
    print(f"Difference: {abs(size2 - size1):,} bytes")


def main():
    """Process and analyze parquet files."""
    # File from the huggingface dataset
    # INPUT_PATH = Path("/Users/ehabba/Downloads/data.parquet")
    INPUT_PATH = Path("data.parquet")

    SAMPLE_OUTPUT = Path("update_data.parquet")
    FULL_OUTPUT = Path("update_data_full.parquet")
    BATCH_SIZE = 1000
    BATCH_LIMIT = 6
    # Process sample with transformation
    process_parquet_file(
        input_path=INPUT_PATH,
        output_path=SAMPLE_OUTPUT,
        batch_limit=BATCH_LIMIT,
        batch_size=BATCH_SIZE,
        process_func=process_df
    )

    # Process larger sample without transformation
    process_parquet_file(
        input_path=INPUT_PATH,
        output_path=FULL_OUTPUT,
        batch_limit=BATCH_LIMIT,
        batch_size=BATCH_SIZE,
    )

    # Compare file sizes
    compare_file_sizes(SAMPLE_OUTPUT, FULL_OUTPUT)


if __name__ == "__main__":
    main()

# uploader.py
import pandas as pd
from huggingface_hub import HfApi

from base_processor import BaseProcessor
from constants import ProcessingConstants


class HFUploader(BaseProcessor):
    def __init__(
            self,
            output_repo: str,
            **kwargs
    ):
        """
        Initialize the HuggingFace uploader.

        Args:
            output_repo (str): HuggingFace repo ID for output files
            **kwargs: Additional arguments passed to BaseProcessor
        """
        super().__init__(**kwargs)
        self.output_repo = output_repo
        self.hf_api = HfApi()
        
        # Set up temp directory
        self.temp_dir = self.data_dir / ProcessingConstants.TEMP_DIR_NAME
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def upload_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """
        Upload a DataFrame to HuggingFace as a parquet file.

        Args:
            df: DataFrame to upload
            name: Name for the output file

        Returns:
            bool: True if upload was successful
        """
        try:
            temp_file = self.temp_dir / f"{name}.parquet"
            df.to_parquet(temp_file, index=False)

            self.hf_api.upload_file(
                path_or_fileobj=str(temp_file),
                path_in_repo=f"{name}.parquet",
                repo_id=self.output_repo,
                repo_type="dataset"
            )

            # self.cleanup(temp_file)
            self.logger.info(f"Successfully uploaded: {name}.parquet")
            return True

        except Exception as e:
            self.logger.error(f"Error uploading {name}: {e}")
            return False

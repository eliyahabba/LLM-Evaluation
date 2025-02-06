
# utils.py
import logging
from typing import Any, Dict

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the project."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = [
        'shots', 'template', 'separator', 'enumerator', 'choices_order'
    ]
    return all(key in config for key in required_keys)
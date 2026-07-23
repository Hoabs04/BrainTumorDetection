"""Entry-point script for dataset preparation."""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_preparer import DatasetPreparer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def main() -> None:
    """Instantiate DatasetPreparer and run the preparation pipeline."""
    preparer = DatasetPreparer()
    preparer.run()


if __name__ == "__main__":
    main()

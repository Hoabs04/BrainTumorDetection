"""Entry-point script for dataset preparation."""

import logging
import sys
from pathlib import Path

from src.data.dataset_builder import DatasetBuilder

# Ensure project root is on sys.path so `src` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_preparer import DatasetPreparer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def main() -> None:
    """Run the complete dataset preparation pipeline."""

    # Step 1: Verify dataset and create directories
    preparer = DatasetPreparer()
    preparer.run()

    # Step 2: Build labeled / unlabeled datasets
    builder = DatasetBuilder()
    builder.run()


if __name__ == "__main__":
    main()

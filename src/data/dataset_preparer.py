"""
Dataset Preparer Module.

Handles initial dataset verification and output directory setup
for the Brain Tumor Detection pipeline.

Responsibilities:
    - Load dataset configuration from YAML.
    - Verify that raw dataset directories exist.
    - Create output directories for downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares the project file system for dataset processing.

    This class performs three core tasks:
        1. Load configuration from ``configs/dataset.yaml``.
        2. Verify that expected raw dataset directories are present.
        3. Create output directories (labeled, unlabeled, processed, metadata).

    It intentionally does **not** copy, resize, split, or read any images.

    Args:
        project_root: Absolute path to the project root directory
            (the folder that contains ``configs/`` and ``datasets/``).
        config_path: Optional override for the YAML config file path.
            Defaults to ``<project_root>/configs/dataset.yaml``.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:

        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        self.config_path: Path = (
            Path(config_path).resolve()
            if config_path is not None
            else self.project_root / "configs" / "dataset.yaml"
        )

        self.config: Dict[str, Any] = {}

        logger.info(
            "DatasetPreparer initialised | root=%s",
            self.project_root
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        """Read and parse the YAML configuration file.

        Returns:
            The parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML content is malformed.
        """
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        logger.info("Configuration loaded from %s", self.config_path)
        return self.config

    def verify_dataset(self) -> None:
        """Verify that all expected raw dataset directories exist.

        Checks for:
            - ``datasets/raw/BrainTumorMRI``
            - ``datasets/raw/Br35H``
            - ``BrainTumorMRI/Training``
            - ``BrainTumorMRI/Testing``
            - ``Br35H/yes``
            - ``Br35H/no``

        Raises:
            FileNotFoundError: With a descriptive message listing the
                first missing directory.
        """
        raw_dir: Path = self.project_root / self.config["dataset"]["raw_dir"]

        # --- BrainTumorMRI paths ---
        btm_name: str = self.config["brain_tumor_mri"]["dataset_name"]
        btm_root: Path = raw_dir / btm_name
        btm_train: Path = btm_root / self.config["brain_tumor_mri"]["training_dir"]
        btm_test: Path = btm_root / self.config["brain_tumor_mri"]["testing_dir"]

        # --- Br35H paths ---
        br35h_name: str = self.config["br35h"]["dataset_name"]
        br35h_root: Path = raw_dir / br35h_name
        br35h_yes: Path = br35h_root / "yes"
        br35h_no: Path = br35h_root / "no"

        required_dirs: Dict[str, Path] = {
            f"Raw dataset root ({btm_name})": btm_root,
            f"Raw dataset root ({br35h_name})": br35h_root,
            f"{btm_name}/Training": btm_train,
            f"{btm_name}/Testing": btm_test,
            f"{br35h_name}/yes": br35h_yes,
            f"{br35h_name}/no": br35h_no,
        }

        for description, dir_path in required_dirs.items():
            if not dir_path.is_dir():
                raise FileNotFoundError(
                    f"Required directory missing — {description}: {dir_path}"
                )
            logger.debug("Verified: %s  →  %s", description, dir_path)

        logger.info(
            "Dataset verification passed  |  all %d directories present",
            len(required_dirs),
        )

    def create_output_directories(self) -> None:
        """Create output directories declared in the configuration.

        Creates (if they do not already exist):
            - ``datasets/labeled``
            - ``datasets/unlabeled``
            - ``datasets/processed``
            - ``datasets/metadata``
        """
        dir_keys = ["labeled_dir", "unlabeled_dir", "processed_dir", "metadata_dir"]

        for key in dir_keys:
            dir_path: Path = self.project_root / self.config["dataset"][key]
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", dir_path)

        logger.info("Output directories created successfully")

    def run(self) -> None:
        """Execute the full preparation pipeline.

        Workflow::

            load_config()  →  verify_dataset()  →  create_output_directories()
        """
        logger.info("=" * 60)
        logger.info("DatasetPreparer — starting preparation pipeline")
        logger.info("=" * 60)

        self.load_config()
        self.verify_dataset()
        self.create_output_directories()

        logger.info("=" * 60)
        logger.info("DatasetPreparer — preparation complete")
        logger.info("=" * 60)

"""
Image Preprocessing Module.

Purpose
-------
Verify dataset quality before training.

This module DOES NOT:
- resize images
- normalize images
- augment images
- modify original images

Those operations belong to transforms.py.

Responsibilities
----------------
- Verify image integrity.
- Detect corrupted images.
- Inspect image properties.
- Collect dataset statistics.
- Generate preprocessing report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Verify dataset quality before training.

    Workflow
    --------
    datasets/labeled
            │
            ▼
    Verify Image
            ▼
    Inspect Properties
            ▼
    Generate Report
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        dataset_config: str | Path | None = None,
    ) -> None:
        """
        Initialize ImagePreprocessor.

        Parameters
        ----------
        project_root:
            Project root directory.

        dataset_config:
            Optional dataset.yaml path.
        """

        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        self.dataset_config_path = (
            Path(dataset_config).resolve()
            if dataset_config
            else self.project_root / "configs" / "dataset.yaml"
        )

        self.dataset_cfg: dict[str, Any] = {}

        # Dataset statistics
        self.total_images = 0
        self.valid_images = 0
        self.corrupted_images = 0

        self.rgb_images = 0
        self.non_rgb_images = 0

        self.small_images = 0

        self.image_formats: dict[str, int] = {}

        # Configuration values
        self.minimum_size = 0
        self.supported_extensions: set[str] = set()

        self.start_time = time.time()

        logger.info(
            "ImagePreprocessor initialized | root=%s",
            self.project_root,
        )
    # ==========================================================
    # Configuration
    # ==========================================================

    def load_config(self) -> None:
        """
        Load configuration from dataset.yaml.

        Expected structure:

        dataset:
            labeled_dir:
            unlabeled_dir:

        preprocessing:
            minimum_size:
            supported_formats:
        """

        if not self.dataset_config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.dataset_config_path}"
            )

        with open(
            self.dataset_config_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.dataset_cfg = yaml.safe_load(file)

        preprocessing_cfg = self.dataset_cfg.get(
            "preprocessing",
            {},
        )

        self.minimum_size = preprocessing_cfg.get(
            "minimum_size",
            64,
        )

        supported_formats = preprocessing_cfg.get(
            "supported_formats",
            [
                "jpg",
                "jpeg",
                "png",
                "bmp",
                "tif",
                "tiff",
            ],
        )

        self.supported_extensions = {
            f".{fmt.lower()}"
            for fmt in supported_formats
        }

        logger.info("=" * 60)
        logger.info("Configuration loaded successfully")
        logger.info("Minimum image size : %d", self.minimum_size)
        logger.info(
            "Supported formats : %s",
            ", ".join(sorted(self.supported_extensions)),
        )
        logger.info("=" * 60)
    # ==========================================================
    # Image Verification
    # ==========================================================

    def verify_image(
        self,
        image_path: Path,
    ) -> bool:
        """
        Verify that an image can be opened.

        Parameters
        ----------
        image_path
            Path to image.

        Returns
        -------
        bool
            True if image is valid.
        """

        try:

            with Image.open(image_path) as image:
                image.verify()

            return True

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            logger.warning(
                "Corrupted image: %s | %s",
                image_path,
                error,
            )

            self.corrupted_images += 1

            return False
    # ----------------------------------------------------------

    def inspect_image(
        self,
        image_path: Path,
    ) -> None:
        """
        Inspect a single image.

        Checks
        ------
        - image integrity
        - image size
        - image mode
        - image format
        """

        self.total_images += 1

        if not self.verify_image(image_path):
            return

        try:

            with Image.open(image_path) as image:

                # ---------- Image format ----------
                image_format = (
                    image.format or "UNKNOWN"
                ).upper()

                self.image_formats[image_format] = (
                    self.image_formats.get(image_format, 0) + 1
                )

                # ---------- Image size ----------
                width, height = image.size

                if (
                    width < self.minimum_size
                    or height < self.minimum_size
                ):

                    logger.warning(
                        "Small image detected: %s (%dx%d)",
                        image_path.name,
                        width,
                        height,
                    )

                    self.small_images += 1

                # ---------- Color mode ----------
                if image.mode == "RGB":

                    self.rgb_images += 1

                else:

                    logger.info(
                        "Non-RGB image: %s (%s)",
                        image_path.name,
                        image.mode,
                    )

                    self.non_rgb_images += 1

            self.valid_images += 1

        except Exception as error:

            logger.error(
                "Failed to inspect image %s | %s",
                image_path,
                error,
            )

            self.corrupted_images += 1
    # ==========================================================
    # Dataset Processing
    # ==========================================================

    def process_directory(
        self,
        directory: Path,
    ) -> None:
        """
        Inspect every image inside a directory.
        """

        if not directory.exists():

            logger.warning(
                "Directory not found: %s",
                directory,
            )

            return

        logger.info("-" * 60)
        logger.info(
            "Scanning directory: %s",
            directory,
        )

        image_files = sorted(

            file

            for file in directory.rglob("*")

            if (
                file.is_file()
                and file.suffix.lower()
                in self.supported_extensions
            )

        )

        logger.info(
            "Found %d image(s)",
            len(image_files),
        )

        for image_path in image_files:

            self.inspect_image(image_path)
    # ----------------------------------------------------------

    def process_labeled(self) -> None:
        """
        Process labeled dataset.

        Structure
        ---------
        datasets/
            labeled/
                train/
                val/
                test/
        """

        labeled_root = (
            self.project_root
            / self.dataset_cfg["dataset"]["labeled_dir"]
        )

        if not labeled_root.exists():
            raise FileNotFoundError(
                f"Labeled dataset not found: {labeled_root}"
            )

        logger.info("=" * 60)
        logger.info("Processing labeled dataset")
        logger.info("=" * 60)

        for split in ("train", "val", "test"):

            split_dir = labeled_root / split

            if not split_dir.exists():

                logger.warning(
                    "Missing split directory: %s",
                    split_dir,
                )

                continue

            logger.info("")
            logger.info("Split: %s", split.upper())

            class_dirs = sorted(

                folder

                for folder in split_dir.iterdir()

                if folder.is_dir()

            )

            for class_dir in class_dirs:

                logger.info(
                    "Processing class: %s",
                    class_dir.name,
                )

                self.process_directory(class_dir)

    # ----------------------------------------------------------

    def process_unlabeled(self) -> None:
        """
        Process unlabeled dataset.

        Structure
        ---------
        datasets/
            unlabeled/
                images/
        """

        unlabeled_root = (
            self.project_root
            / self.dataset_cfg["dataset"]["unlabeled_dir"]
        )

        if not unlabeled_root.exists():

            logger.warning(
                "Unlabeled dataset directory not found."
            )

            return

        image_dir = unlabeled_root / "images"

        if not image_dir.exists():

            logger.warning(
                "Missing directory: %s",
                image_dir,
            )

            return

        logger.info("=" * 60)
        logger.info("Processing unlabeled dataset")
        logger.info("=" * 60)

        self.process_directory(image_dir)
    # ==========================================================
    # Report
    # ==========================================================

    def generate_report(self) -> None:
        """
        Generate preprocessing report.
        """

        report_dir = (
            self.project_root
            / "reports"
            / "preprocessing"
        )

        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        elapsed_time = round(
            time.time() - self.start_time,
            2,
        )

        report = {

            "status": "success",

            "total_images": self.total_images,

            "valid_images": self.valid_images,

            "corrupted_images": self.corrupted_images,

            "rgb_images": self.rgb_images,

            "non_rgb_images": self.non_rgb_images,

            "small_images": self.small_images,

            "minimum_size": self.minimum_size,

            "supported_formats": sorted(
                self.supported_extensions
            ),

            "image_formats": self.image_formats,

            "processing_time_seconds": elapsed_time,

        }

        report_path = (
            report_dir
            / "preprocessing_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        logger.info("=" * 60)
        logger.info("Report saved: %s", report_path)
        logger.info("=" * 60)

    # ==========================================================
    # Pipeline
    # ==========================================================

    def run(self) -> None:
        """
        Execute preprocessing pipeline.
        """

        logger.info("=" * 60)
        logger.info("IMAGE PREPROCESSING STARTED")
        logger.info("=" * 60)

        self.load_config()

        self.process_labeled()

        self.process_unlabeled()

        self.generate_report()

        logger.info("=" * 60)
        logger.info("IMAGE PREPROCESSING FINISHED")
        logger.info("=" * 60)

        logger.info("Total Images     : %d", self.total_images)
        logger.info("Valid Images     : %d", self.valid_images)
        logger.info("Corrupted Images : %d", self.corrupted_images)
        logger.info("RGB Images       : %d", self.rgb_images)
        logger.info("Non-RGB Images   : %d", self.non_rgb_images)
        logger.info("Small Images     : %d", self.small_images)


# ==============================================================
# Standalone execution
# ==============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ImagePreprocessor().run()

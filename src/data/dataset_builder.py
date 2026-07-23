"""
Dataset Builder Module.

Organises raw datasets into a structured labeled / unlabeled layout
ready for downstream training and evaluation.

Responsibilities:
    - Split BrainTumorMRI Training data into train / val sets.
    - Copy BrainTumorMRI Testing data into a dedicated test folder.
    - Aggregate Br35H images as an unlabeled pool.
    - Generate metadata files (classes.json, dataset_info.json, split.csv).
    - Generate a build report (build_report.json).

This module does **not** resize, normalise, augment, or read pixel data.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build the organised dataset layout from raw sources.

    Args:
        project_root: Absolute path to the project root directory.
            Defaults to two levels above this file.
        config_path: Optional override for the YAML configuration file.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

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

        # Bookkeeping for metadata generation
        self._split_records: List[Dict[str, str]] = []

        logger.info(
            "DatasetBuilder initialised | root=%s", self.project_root
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        """Load and parse ``configs/dataset.yaml``.

        Returns:
            The parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the config file is missing.
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

    def clean_output_dirs(self) -> None:
        """Remove existing output directories before a fresh build.

        Removes (with ``ignore_errors=True``):
            - ``datasets/labeled``
            - ``datasets/unlabeled``
            - ``datasets/metadata``

        This ensures no stale data leaks between runs.
        """
        cfg_ds = self.config["dataset"]

        dirs_to_clean = [
            self.project_root / cfg_ds["labeled_dir"],
            self.project_root / cfg_ds["unlabeled_dir"],
            self.project_root / cfg_ds["metadata_dir"],
        ]

        for dir_path in dirs_to_clean:
            shutil.rmtree(dir_path, ignore_errors=True)
            logger.info("Cleaned: %s", dir_path)

        logger.info("Output directories cleaned")

    def prepare_output_dirs(self) -> None:
        """Create all output directories declared in the configuration.

        Directories created:
            - ``datasets/labeled/train/<class>``
            - ``datasets/labeled/val/<class>``
            - ``datasets/labeled/test/<class>``
            - ``datasets/unlabeled/images``
            - ``datasets/metadata``
        """
        labeled_dir = self.project_root / self.config["dataset"]["labeled_dir"]
        unlabeled_dir = self.project_root / self.config["dataset"]["unlabeled_dir"]
        metadata_dir = self.project_root / self.config["dataset"]["metadata_dir"]

        classes: List[str] = self.config["brain_tumor_mri"]["classes"]

        for split in ("train", "val", "test"):
            for cls in classes:
                (labeled_dir / split / cls).mkdir(parents=True, exist_ok=True)

        (unlabeled_dir / "images").mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Output directories prepared")

    def split_brain_tumor_mri(self) -> Tuple[int, int]:
        """Split BrainTumorMRI/Training into train and val sets.

        The split is stratified per class and reproducible via the
        ``random_seed`` value in the configuration.

        Returns:
            A tuple ``(train_count, val_count)``.
        """
        logger.info("Building BrainTumorMRI training dataset...")

        cfg_ds = self.config["dataset"]
        cfg_btm = self.config["brain_tumor_mri"]
        cfg_split = self.config["split"]

        raw_training_dir: Path = (
            self.project_root
            / cfg_ds["raw_dir"]
            / cfg_btm["dataset_name"]
            / cfg_btm["training_dir"]
        )
        labeled_dir: Path = self.project_root / cfg_ds["labeled_dir"]
        extensions = self._supported_extensions()
        seed: int = cfg_split["random_seed"]
        train_ratio: float = cfg_split["train_ratio"]
        dataset_name: str = cfg_btm["dataset_name"]

        classes: List[str] = cfg_btm["classes"]
        total_train = 0
        total_val = 0

        for cls in classes:
            class_dir = raw_training_dir / cls
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory missing: {class_dir}"
                )

            images = self._collect_images(class_dir, extensions)
            rng = random.Random(seed)
            rng.shuffle(images)

            split_idx = int(len(images) * train_ratio)
            train_files = images[:split_idx]
            val_files = images[split_idx:]

            for fp in train_files:
                dst = labeled_dir / "train" / cls / fp.name
                shutil.copy2(fp, dst)
                self._split_records.append({
                    "filename": fp.name,
                    "filepath": str(dst.relative_to(self.project_root)),
                    "dataset": dataset_name,
                    "split": "train",
                    "class": cls,
                })

            for fp in val_files:
                dst = labeled_dir / "val" / cls / fp.name
                shutil.copy2(fp, dst)
                self._split_records.append({
                    "filename": fp.name,
                    "filepath": str(dst.relative_to(self.project_root)),
                    "dataset": dataset_name,
                    "split": "val",
                    "class": cls,
                })

            total_train += len(train_files)
            total_val += len(val_files)

            logger.info(
                "Class %-12s | train=%4d  val=%4d",
                cls, len(train_files), len(val_files),
            )

        logger.info(
            "BrainTumorMRI split complete | train=%d  val=%d",
            total_train, total_val,
        )
        return total_train, total_val

    def copy_testing_set(self) -> int:
        """Copy BrainTumorMRI/Testing into ``datasets/labeled/test``.

        Returns:
            The total number of images copied.
        """
        logger.info("Building BrainTumorMRI testing dataset...")

        cfg_ds = self.config["dataset"]
        cfg_btm = self.config["brain_tumor_mri"]

        raw_testing_dir: Path = (
            self.project_root
            / cfg_ds["raw_dir"]
            / cfg_btm["dataset_name"]
            / cfg_btm["testing_dir"]
        )
        labeled_dir: Path = self.project_root / cfg_ds["labeled_dir"]
        extensions = self._supported_extensions()
        dataset_name: str = cfg_btm["dataset_name"]

        classes: List[str] = cfg_btm["classes"]
        total = 0

        for cls in classes:
            class_dir = raw_testing_dir / cls
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected test class directory missing: {class_dir}"
                )

            images = self._collect_images(class_dir, extensions)

            for fp in images:
                dst = labeled_dir / "test" / cls / fp.name
                shutil.copy2(fp, dst)
                self._split_records.append({
                    "filename": fp.name,
                    "filepath": str(dst.relative_to(self.project_root)),
                    "dataset": dataset_name,
                    "split": "test",
                    "class": cls,
                })

            total += len(images)
            logger.info("Test class %-12s | images=%4d", cls, len(images))

        logger.info("Testing set copied | total=%d", total)
        return total

    def prepare_unlabeled_dataset(self) -> int:
        """Copy Br35H images into ``datasets/unlabeled/images``.

        Ignores directories listed under ``br35h.ignore`` in the config.
        Handles filename collisions by appending an incremental suffix
        (e.g. ``image.jpg`` → ``image_1.jpg``).

        Returns:
            The total number of images copied.
        """
        logger.info("Preparing Br35H unlabeled dataset...")

        cfg_ds = self.config["dataset"]
        cfg_br35h = self.config["br35h"]

        br35h_root: Path = (
            self.project_root
            / cfg_ds["raw_dir"]
            / cfg_br35h["dataset_name"]
        )
        unlabeled_images_dir: Path = (
            self.project_root / cfg_ds["unlabeled_dir"] / "images"
        )
        extensions = self._supported_extensions()
        use_dirs: List[str] = cfg_br35h["use_unlabeled"]
        dataset_name: str = cfg_br35h["dataset_name"]

        total = 0

        for folder_name in use_dirs:
            src_dir = br35h_root / folder_name
            if not src_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected Br35H subdirectory missing: {src_dir}"
                )

            images = self._collect_images(src_dir, extensions)

            for fp in images:
                dst = self._resolve_collision(unlabeled_images_dir, fp.name)
                shutil.copy2(fp, dst)
                self._split_records.append({
                    "filename": dst.name,
                    "filepath": str(dst.relative_to(self.project_root)),
                    "dataset": dataset_name,
                    "split": "unlabeled",
                    "class": "",
                })

            total += len(images)
            logger.info(
                "Br35H/%s | images=%4d", folder_name, len(images)
            )

        logger.info("Unlabeled dataset prepared | total=%d", total)
        return total

    def generate_metadata(self) -> None:
        """Write metadata files to ``datasets/metadata/``.

        Files generated:
            - ``classes.json``  – ordered list of class names.
            - ``dataset_info.json`` – image counts and class summary.
            - ``split.csv`` – per-image record with filename, filepath,
              dataset, split, class.
            - ``build_report.json`` – concise build summary.
        """
        logger.info("Generating metadata...")

        metadata_dir: Path = (
            self.project_root / self.config["dataset"]["metadata_dir"]
        )
        classes: List[str] = self.config["brain_tumor_mri"]["classes"]

        # ---- classes.json ----
        classes_path = metadata_dir / "classes.json"
        with open(classes_path, "w", encoding="utf-8") as fh:
            json.dump(classes, fh, indent=4, ensure_ascii=False)
        logger.info("Written %s", classes_path)

        # ---- dataset_info.json ----
        counts = {"train": 0, "val": 0, "test": 0, "unlabeled": 0}
        for rec in self._split_records:
            counts[rec["split"]] += 1

        info = {
            "num_train_images": counts["train"],
            "num_validation_images": counts["val"],
            "num_testing_images": counts["test"],
            "num_unlabeled_images": counts["unlabeled"],
            "num_classes": len(classes),
            "class_names": classes,
        }
        info_path = metadata_dir / "dataset_info.json"
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=4, ensure_ascii=False)
        logger.info("Written %s", info_path)

        # ---- split.csv ----
        csv_path = metadata_dir / "split.csv"
        fieldnames = ["filename", "filepath", "dataset", "split", "class"]
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._split_records)
        logger.info("Written %s  (%d rows)", csv_path, len(self._split_records))

        # ---- build_report.json ----
        total = sum(counts.values())
        report = {
            "status": "success",
            "brain_tumor_train": counts["train"],
            "brain_tumor_validation": counts["val"],
            "brain_tumor_test": counts["test"],
            "br35h_unlabeled": counts["unlabeled"],
            "total_images": total,
        }
        report_path = metadata_dir / "build_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=4, ensure_ascii=False)
        logger.info("Written %s", report_path)

        logger.info("Metadata generation complete")

    def run(self) -> None:
        """Execute the full dataset-building pipeline.

        Workflow::

            load_config()
            clean_output_dirs()
            prepare_output_dirs()
            split_brain_tumor_mri()
            copy_testing_set()
            prepare_unlabeled_dataset()
            generate_metadata()
        """
        logger.info("=" * 60)
        logger.info("DatasetBuilder — starting build pipeline")
        logger.info("=" * 60)

        self.load_config()
        self.clean_output_dirs()
        self.prepare_output_dirs()
        self.split_brain_tumor_mri()
        self.copy_testing_set()
        self.prepare_unlabeled_dataset()
        self.generate_metadata()

        logger.info("=" * 60)
        logger.info("Dataset build completed successfully.")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _supported_extensions(self) -> Tuple[str, ...]:
        """Return a tuple of supported file extensions with leading dots."""
        return tuple(
            f".{ext.lower()}" for ext in self.config["supported_extensions"]
        )

    @staticmethod
    def _collect_images(
        directory: Path,
        extensions: Tuple[str, ...],
    ) -> List[Path]:
        """Gather image paths from *directory* matching *extensions*.

        Args:
            directory: Folder to scan (non-recursive).
            extensions: Accepted file suffixes (e.g. ``('.jpg', '.png')``).

        Returns:
            A sorted list of matching file paths.
        """
        images = sorted(
            fp for fp in directory.iterdir()
            if fp.is_file() and fp.suffix.lower() in extensions
        )
        return images

    @staticmethod
    def _resolve_collision(target_dir: Path, filename: str) -> Path:
        """Return a unique file path inside *target_dir*.

        If ``target_dir/filename`` already exists, an incremental suffix
        is appended before the extension::

            image.jpg  →  image_1.jpg  →  image_2.jpg  …

        Args:
            target_dir: Destination directory.
            filename: Original file name.

        Returns:
            A ``Path`` that does not yet exist.
        """
        candidate = target_dir / filename
        if not candidate.exists():
            return candidate

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        counter = 1

        while True:
            candidate = target_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

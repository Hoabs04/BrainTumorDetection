"""
PyTorch DataLoader Factory.

Build datasets and dataloaders for:

- Train
- Validation
- Test
- Unlabeled

This module connects:

Dataset
    ↓
Transforms
    ↓
ImageFolder
    ↓
DataLoader
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.transforms import TransformFactory
from src.data.unlabeled_dataset import UnlabeledDataset

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating PyTorch datasets and dataloaders.
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

        self.config_path = (
            Path(config_path).resolve()
            if config_path
            else self.project_root / "configs" / "dataset.yaml"
        )

        self.config: dict[str, Any] = {}

        # Dataset paths
        self.train_dir: Path | None = None
        self.val_dir: Path | None = None
        self.test_dir: Path | None = None
        self.unlabeled_dir: Path | None = None

        # Loader configuration
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True
        self.drop_last = False

        # Transform factory
        self.transform_factory = TransformFactory(
            project_root=self.project_root
        )

        # Dataset objects
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.unlabeled_dataset = None

        # DataLoader objects
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.unlabeled_loader = None

        logger.info(
            "DataLoaderFactory initialized | root=%s",
            self.project_root,
        )

    # ==========================================================
    # Configuration
    # ==========================================================

    def load_config(self) -> None:
        """
        Load dataset and DataLoader configuration.
        """

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = yaml.safe_load(file)

        # ------------------------------------------------------
        # Dataset paths
        # ------------------------------------------------------

        dataset_cfg = self.config.get(
            "dataset",
            {},
        )

        labeled_root = (
            self.project_root
            / dataset_cfg.get(
                "labeled_dir",
                "datasets/labeled",
            )
        )

        self.train_dir = labeled_root / "train"
        self.val_dir = labeled_root / "val"
        self.test_dir = labeled_root / "test"

        self.unlabeled_dir = (
            self.project_root
            / dataset_cfg.get(
                "unlabeled_dir",
                "datasets/unlabeled",
            )
            / "images"
        )

        # ------------------------------------------------------
        # Loader configuration
        # ------------------------------------------------------

        loader_cfg = self.config.get(
            "loader",
            {},
        )

        self.batch_size = loader_cfg.get(
            "batch_size",
            32,
        )

        self.num_workers = loader_cfg.get(
            "num_workers",
            4,
        )

        self.pin_memory = loader_cfg.get(
            "pin_memory",
            True,
        )

        self.shuffle = loader_cfg.get(
            "shuffle",
            True,
        )

        self.drop_last = loader_cfg.get(
            "drop_last",
            False,
        )

        self.transform_factory.load_config()

        logger.info("=" * 60)
        logger.info("Loader configuration loaded")
        logger.info("Train      : %s", self.train_dir)
        logger.info("Validation : %s", self.val_dir)
        logger.info("Test       : %s", self.test_dir)
        logger.info("Batch Size : %d", self.batch_size)
        logger.info("Workers    : %d", self.num_workers)
        logger.info("=" * 60)

    # ==========================================================
    # Dataset Builder
    # ==========================================================

    def build_datasets(self) -> None:
        """
        Build PyTorch datasets.
        """

        logger.info("=" * 60)
        logger.info("Building datasets...")
        logger.info("=" * 60)

        train_transform = self.transform_factory.build(
            train=True,
        )

        evaluation_transform = self.transform_factory.build(
            train=False,
        )

        # ------------------------------------------------------
        # Train dataset
        # ------------------------------------------------------

        if self.train_dir is None or not self.train_dir.exists():
            raise FileNotFoundError(
                f"Train dataset not found: {self.train_dir}"
            )

        self.train_dataset = ImageFolder(
            root=self.train_dir,
            transform=train_transform,
        )

        logger.info(
            "Train dataset: %d images | %d classes",
            len(self.train_dataset),
            len(self.train_dataset.classes),
        )

        # ------------------------------------------------------
        # Validation dataset
        # ------------------------------------------------------

        if self.val_dir is None or not self.val_dir.exists():
            raise FileNotFoundError(
                f"Validation dataset not found: {self.val_dir}"
            )

        self.val_dataset = ImageFolder(
            root=self.val_dir,
            transform=evaluation_transform,
        )

        logger.info(
            "Validation dataset: %d images",
            len(self.val_dataset),
        )

        # ------------------------------------------------------
        # Test dataset
        # ------------------------------------------------------

        if self.test_dir is None or not self.test_dir.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {self.test_dir}"
            )

        self.test_dataset = ImageFolder(
            root=self.test_dir,
            transform=evaluation_transform,
        )

        logger.info(
            "Test dataset: %d images",
            len(self.test_dataset),
        )

        # ------------------------------------------------------
        # Unlabeled dataset
        # ------------------------------------------------------

        if (
            self.unlabeled_dir is not None
            and self.unlabeled_dir.exists()
        ):

            self.unlabeled_dataset = UnlabeledDataset(
                root=self.unlabeled_dir,
                transform=evaluation_transform,
            )

            logger.info(
                "Unlabeled dataset: %d images",
                len(self.unlabeled_dataset),
            )

        else:

            logger.warning(
                "Unlabeled dataset not found: %s",
                self.unlabeled_dir,
            )

        logger.info("=" * 60)
        logger.info("All datasets created successfully.")
        logger.info("=" * 60)

    # ==========================================================
    # DataLoader Builder
    # ==========================================================

    def build_loaders(self) -> None:
        """
        Build PyTorch DataLoaders.
        """

        logger.info("=" * 60)
        logger.info("Building dataloaders...")
        logger.info("=" * 60)

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        if self.unlabeled_dataset is not None:

            self.unlabeled_loader = DataLoader(
                dataset=self.unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        logger.info("DataLoaders created successfully.")

    def get_train_loader(self) -> DataLoader:
        return self.train_loader


    def get_val_loader(self) -> DataLoader:
        return self.val_loader


    def get_test_loader(self) -> DataLoader:
        return self.test_loader


    def get_unlabeled_loader(self) -> DataLoader | None:
        return self.unlabeled_loader

    # ==========================================================
    # Utilities
    # ==========================================================

    def summary(self) -> None:
        """
        Print DataLoader summary.
        """

        logger.info("=" * 60)
        logger.info("DataLoader Summary")
        logger.info("=" * 60)

        if self.train_dataset is not None:
            logger.info("Train      : %d", len(self.train_dataset))

        if self.val_dataset is not None:
            logger.info("Validation : %d", len(self.val_dataset))

        if self.test_dataset is not None:
            logger.info("Test       : %d", len(self.test_dataset))

        if self.unlabeled_dataset is not None:
            logger.info("Unlabeled  : %d", len(self.unlabeled_dataset))

        logger.info("Batch Size : %d", self.batch_size)
        logger.info("Workers    : %d", self.num_workers)
        logger.info("=" * 60)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    factory = DataLoaderFactory()

    factory.load_config()

    factory.build_datasets()

    factory.build_loaders()

    factory.summary()
"""
Image Transform Factory.

Create torchvision transform pipelines for:

- Training
- Evaluation (Validation/Test/Inference)

All configurations are loaded from configs/dataset.yaml.

This module never loads images or datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from torchvision import transforms
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)


class TransformFactory:
    """
    Factory for building torchvision transform pipelines.
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

        self.image_size = 224
        self.interpolation = InterpolationMode.BILINEAR

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.horizontal_flip = 0.5
        self.rotation = 15

        self.color_jitter = {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        }

        logger.info(
            "TransformFactory initialized | root=%s",
            self.project_root,
        )

    # ==========================================================
    # Configuration
    # ==========================================================

    def load_config(self) -> None:
        """
        Load transform configuration from dataset.yaml.
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

        transform_cfg = self.config.get(
            "transforms",
            {},
        )

        self.image_size = transform_cfg.get(
            "image_size",
            224,
        )

        interpolation_name = (
            transform_cfg.get(
                "interpolation",
                "bilinear",
            )
            .lower()
            .strip()
        )

        interpolation_map = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }

        self.interpolation = interpolation_map.get(
            interpolation_name,
            InterpolationMode.BILINEAR,
        )

        normalize_cfg = transform_cfg.get(
            "normalize",
            {},
        )

        self.mean = normalize_cfg.get(
            "mean",
            [0.485, 0.456, 0.406],
        )

        self.std = normalize_cfg.get(
            "std",
            [0.229, 0.224, 0.225],
        )

        augmentation_cfg = self.config.get(
            "augmentation",
            {},
        )

        self.horizontal_flip = augmentation_cfg.get(
            "horizontal_flip",
            0.5,
        )

        self.rotation = augmentation_cfg.get(
            "rotation",
            15,
        )

        self.color_jitter = augmentation_cfg.get(
            "color_jitter",
            {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
        )

        logger.info("=" * 60)
        logger.info("Transform configuration loaded")
        logger.info("Image size      : %d", self.image_size)
        logger.info("Interpolation   : %s", interpolation_name)
        logger.info("Normalize mean  : %s", self.mean)
        logger.info("Normalize std   : %s", self.std)
        logger.info("=" * 60)

    # ==========================================================
    # Transform Builder
    # ==========================================================

    def build(
        self,
        train: bool = False,
    ) -> transforms.Compose:
        """
        Build transform pipeline.

        Parameters
        ----------
        train
            If True, apply data augmentation.
        """

        logger.info(
            "Building %s transform...",
            "training" if train else "evaluation",
        )

        pipeline = [

            transforms.Resize(
                (
                    self.image_size,
                    self.image_size,
                ),
                interpolation=self.interpolation,
            ),

            transforms.Lambda(
                lambda img: img.convert("RGB")
            ),

        ]

        if train:

            pipeline.extend(

                [

                    transforms.RandomHorizontalFlip(
                        p=self.horizontal_flip,
                    ),

                    transforms.RandomRotation(
                        degrees=self.rotation,
                    ),

                    transforms.ColorJitter(
                        brightness=self.color_jitter["brightness"],
                        contrast=self.color_jitter["contrast"],
                        saturation=self.color_jitter["saturation"],
                        hue=self.color_jitter["hue"],
                    ),

                ]

            )

        pipeline.extend(

            [

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),

            ]

        )

        return transforms.Compose(pipeline)

    # ==========================================================
    # Utilities
    # ==========================================================

    def summary(self) -> None:
        """
        Print current transform configuration.
        """

        logger.info("=" * 60)
        logger.info("Transform Factory Summary")
        logger.info("=" * 60)

        logger.info(
            "Image Size        : %d",
            self.image_size,
        )

        logger.info(
            "Interpolation     : %s",
            self.interpolation,
        )

        logger.info(
            "Normalize Mean    : %s",
            self.mean,
        )

        logger.info(
            "Normalize Std     : %s",
            self.std,
        )

        logger.info(
            "Horizontal Flip   : %.2f",
            self.horizontal_flip,
        )

        logger.info(
            "Rotation          : %d",
            self.rotation,
        )

        logger.info(
            "Color Jitter      : %s",
            self.color_jitter,
        )

        logger.info("=" * 60)

# ==============================================================
# Standalone execution
# ==============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    factory = TransformFactory()

    factory.load_config()

    factory.summary()

    train_transform = factory.build(train=True)

    evaluation_transform = factory.build(train=False)

    logger.info("Train Transform:")
    logger.info(train_transform)

    logger.info("")

    logger.info("Evaluation Transform:")
    logger.info(evaluation_transform)
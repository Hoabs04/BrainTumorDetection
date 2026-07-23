"""
Unlabeled Dataset.

Custom PyTorch Dataset for loading unlabeled images.

Unlike torchvision.datasets.ImageFolder,
this dataset does NOT require class folders.

Directory structure
-------------------

datasets/

    unlabeled/

        images/

            img001.jpg

            img002.jpg

            ...

Returns
-------

(image, image_path)
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class UnlabeledDataset(Dataset):
    """
    Dataset for unlabeled images.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        root: str | Path,
        transform=None,
    ) -> None:

        self.root = Path(root).resolve()

        if not self.root.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.root}"
            )

        self.transform = transform

        self.image_paths = []

        self.scan_images()

        logger.info(
            "UnlabeledDataset initialized (%d images)",
            len(self.image_paths),
        )

    # ==========================================================
    # Image Scanning
    # ==========================================================

    def scan_images(self) -> None:
        """
        Scan all supported images recursively.
        """

        logger.info("=" * 60)
        logger.info("Scanning unlabeled dataset...")
        logger.info("Root: %s", self.root)

        self.image_paths = sorted(

            file

            for file in self.root.rglob("*")

            if (
                file.is_file()
                and file.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
            )

        )

        if not self.image_paths:

            logger.warning(
                "No images found in %s",
                self.root,
            )

        logger.info(
            "Found %d image(s)",
            len(self.image_paths),
        )

        logger.info("=" * 60)

    # ==========================================================
    # PyTorch Dataset Interface
    # ==========================================================

    def __len__(self) -> int:
        """
        Return the number of images.
        """

        return len(self.image_paths)

    # ----------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ):
        """
        Get one image.

        Returns
        -------
        tuple
            (image, image_path)
        """

        image_path = self.image_paths[index]

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as error:

            logger.error(
                "Cannot open image: %s | %s",
                image_path,
                error,
            )

            raise

        if self.transform is not None:

            image = self.transform(image)

        return image, str(image_path)

    # ==========================================================
    # Utilities
    # ==========================================================

    def summary(self) -> None:
        """
        Print dataset summary.
        """

        logger.info("=" * 60)
        logger.info("Unlabeled Dataset Summary")
        logger.info("=" * 60)

        logger.info(
            "Root Directory : %s",
            self.root,
        )

        logger.info(
            "Total Images   : %d",
            len(self.image_paths),
        )

        logger.info(
            "Transform      : %s",
            self.transform is not None,
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

    dataset = UnlabeledDataset(
        root="datasets/unlabeled/images",
    )

    dataset.summary()

    logger.info(
        "Dataset Length: %d",
        len(dataset),
    )

    if len(dataset) > 0:

        image, image_path = dataset[0]

        logger.info(
            "First Image : %s",
            image_path,
        )

        logger.info(
            "Image Size  : %s",
            image.size,
        )
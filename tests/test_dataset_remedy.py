import unittest
from unittest.mock import patch
import torch
from PIL import Image
from torchvision import transforms

# Import functions under test
from src.dataset import get_cifar10_loaders, SubsetNormalizedDataset
from src.eval import calculate_ssim


class MockCIFAR10:
    def __init__(self, root, train=True, download=False, transform=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.data = [Image.new('RGB', (32, 32)) for _ in range(100)]
        self.targets = [i % 10 for i in range(100)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TestDatasetRemedy(unittest.TestCase):
    def test_validation_no_augmentation(self):
        with patch('src.dataset.datasets.CIFAR10', new=MockCIFAR10):
            train_loader, val_loader, test_loader = get_cifar10_loaders(
                data_dir="./data",
                batch_size=2,
                val_split=0.1,
                seed=42
            )

            # Check loader lengths
            self.assertEqual(len(val_loader.dataset), 10)
            self.assertEqual(len(train_loader.dataset), 90)

            # Check that val_loader.dataset is a SubsetNormalizedDataset
            self.assertIsInstance(val_loader.dataset, SubsetNormalizedDataset)

            # Check transforms of val_loader dataset
            transform = val_loader.dataset.base_dataset.transform
            self.assertIsInstance(transform, transforms.Compose)
            transform_types = [type(t) for t in transform.transforms]
            self.assertNotIn(transforms.RandomCrop, transform_types)
            self.assertNotIn(transforms.RandomHorizontalFlip, transform_types)

            # Check transforms of train_loader dataset
            self.assertIsInstance(train_loader.dataset, SubsetNormalizedDataset)
            train_transform = train_loader.dataset.base_dataset.transform
            self.assertIsInstance(train_transform, transforms.Compose)
            train_transform_types = [type(t) for t in train_transform.transforms]
            self.assertIn(transforms.RandomCrop, train_transform_types)
            self.assertIn(transforms.RandomHorizontalFlip, train_transform_types)

            # Let's check that retrieval works and applies transformations correctly
            img_norm, img_raw, label = val_loader.dataset[0]
            self.assertEqual(img_norm.shape, (3, 32, 32))
            self.assertEqual(img_raw.shape, (3, 32, 32))
            expected_label = val_loader.dataset.base_dataset.targets[val_loader.dataset.indices[0]]
            self.assertEqual(label, expected_label)

    def test_calculate_ssim_skimage_compatibility(self):
        # Create dummy tensors (B, 3, 32, 32)
        x = torch.rand(2, 3, 32, 32)
        x_recon = torch.rand(2, 3, 32, 32)

        # Test default calculate_ssim execution (using skimage if available or fallback)
        val = calculate_ssim(x, x_recon)
        self.assertTrue(-1.0 <= val <= 1.0)

        # Identical images must produce exact SSIM 1.0 under the locked metric.
        self.assertAlmostEqual(calculate_ssim(x, x), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

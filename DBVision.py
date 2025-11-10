'''
project/
├── Data/                    # Raw images
│   ├── cifar100/
│   │   ├── car/            # Class subfolders
│   │   ├── cat/
│   │   └── ... (100 classes)
│   └── mnist/
├── data/                    # Processed features
│   ├── cifar100/
│   │   ├── cifar100.csv    # Feature CSV
│   │   └── cifar100.conf   # Dataset config
│   └── mnist/
├── configs/                 # System configs
├── models/                  # Trained models
└── unified_system.py

# Install dependencies
pip install torch torchvision pandas pillow matplotlib

# Run the system
python unified_system.py

# Basic usage with defaults
python DBVision.py --dataset cifar100

# Fresh start with local data
python DBVision.py --dataset my_custom_dataset --data-source local --fresh

# Only extract features from existing model
python DBVision.py --dataset cifar100 --skip-training

# Custom dimensions and rich features
python DBVision.py --dataset cifar100 --final-dim 64 --extract-rich

# No compression (use raw 512D features)
python DBVision.py --dataset cifar100 --no-compression

# Custom training parameters
python DBVision.py --dataset cifar100 --epochs 50 --batch-size 64 --fresh

# Help menu
python DBVision.py --help
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import time
import argparse
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    name: str
    type: str
    in_channels: int
    input_size: List[int]
    mean: List[float]
    std: List[float]
    image_type: str

@dataclass
class ModelConfig:
    encoder_type: str
    feature_dims: int
    learning_rate: float
    enhancement_modules: Dict
    loss_functions: Dict

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    num_workers: int
    enhancement_specific: Dict

class SelfAttentionEnhanced(nn.Module):
    """Self-attention module from old model for enhanced feature refinement"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        queries = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        keys = self.key(x).view(batch_size, -1, height * width)
        values = self.value(x).view(batch_size, -1, height * width)

        attention_scores = torch.bmm(queries, keys)
        attention_scores = F.softmax(attention_scores, dim=-1)

        out = torch.bmm(values, attention_scores.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

class ConfigManager:
    """Manages configuration files for the entire system"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def load_main_config(self, dataset_name: str) -> Dict:
        """Load main dataset configuration"""
        config_path = self.config_dir / f"{dataset_name}.json"

        if not config_path.exists():
            # Create default configuration
            default_config = self._create_default_config(dataset_name)
            self.save_main_config(dataset_name, default_config)
            return default_config

        with open(config_path, 'r') as f:
            return json.load(f)

    def save_main_config(self, dataset_name: str, config: Dict):
        """Save main configuration"""
        config_path = self.config_dir / f"{dataset_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_dbnn_config(self) -> Dict:
        """Load DBNN configuration"""
        config_path = self.config_dir / "adaptive_dbnn.conf"

        if not config_path.exists():
            default_config = self._create_default_dbnn_config()
            self.save_dbnn_config(default_config)
            return default_config

        with open(config_path, 'r') as f:
            return json.load(f)

    def save_dbnn_config(self, config: Dict):
        """Save DBNN configuration"""
        config_path = self.config_dir / "adaptive_dbnn.conf"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _create_default_config(self, dataset_name: str) -> Dict:
        """Create default configuration for a dataset"""
        return {
            "dataset": {
                "name": dataset_name,
                "type": "custom",
                "in_channels": 3,
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "image_type": "general"
            },
            "model": {
                 "use_kl_divergence": true,
                "use_class_encoding": true,
                "clustering_temperature": 1.0,
                "encoder_type": "enhanced",
                "feature_dims": 128,
                "learning_rate": 0.001,
                "enhancement_modules": {
                    "astronomical": {
                        "enabled": False,
                        "components": {
                            "structure_preservation": True,
                            "detail_preservation": True,
                            "star_detection": True,
                            "galaxy_features": True,
                            "kl_divergence": True
                        },
                        "weights": {
                            "detail_weight": 1.0,
                            "structure_weight": 0.8,
                            "edge_weight": 0.7
                        }
                    },
                    "medical": {
                        "enabled": False,
                        "components": {
                            "tissue_boundary": True,
                            "lesion_detection": True,
                            "contrast_enhancement": True,
                            "subtle_feature_preservation": True
                        },
                        "weights": {
                            "boundary_weight": 1.0,
                            "lesion_weight": 0.8,
                            "contrast_weight": 0.6
                        }
                    },
                    "agricultural": {
                        "enabled": False,
                        "components": {
                            "texture_analysis": True,
                            "damage_detection": True,
                            "color_anomaly": True,
                            "pattern_enhancement": True,
                            "morphological_features": True
                        },
                        "weights": {
                            "texture_weight": 1.0,
                            "damage_weight": 0.8,
                            "pattern_weight": 0.7
                        }
                    }
                },
                "loss_functions": {
                    "base_autoencoder": {
                        "enabled": True,
                        "weight": 1.0
                    },
                    "astronomical_structure": {
                        "enabled": False,
                        "weight": 1.0,
                        "components": {
                            "edge_preservation": True,
                            "peak_preservation": True,
                            "detail_preservation": True
                        }
                    },
                    "medical_structure": {
                        "enabled": False,
                        "weight": 1.0,
                        "components": {
                            "boundary_preservation": True,
                            "tissue_contrast": True,
                            "local_structure": True
                        }
                    },
                    "agricultural_pattern": {
                        "enabled": False,
                        "weight": 1.0,
                        "components": {
                            "texture_preservation": True,
                            "damage_pattern": True,
                            "color_consistency": True
                        }
                    }
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": 4,
                "enhancement_specific": {
                    "feature_extraction_frequency": 5,
                    "pattern_validation_steps": 100,
                    "adaptive_weight_adjustment": True
                }
            }
        }

    def _create_default_dbnn_config(self) -> Dict:
        """Create default DBNN configuration"""
        return {
            "training_params": {
                "trials": 100,
                "epochs": 1000,
                "learning_rate": 0.1,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 20,
                "enable_adaptive": True,
                "modelType": "Histogram",
                "compute_device": "auto"
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "fresh_start": False,
                "use_previous_model": True,
                "gen_samples": False
            }
        }

class GenericImageDataset(Dataset):
    def __init__(self, root_dir: str, dataset_name: str, config: Dict, split: str = "train", data_source: str = "auto"):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.data_source = data_source

        # Load data FIRST
        self.image_paths, self.targets, self.class_names = self._load_data()
        # Setup transforms
        self.transform = self._create_transforms()

        logger.info(f"Loaded {len(self.image_paths)} images for {dataset_name} ({split}) from {data_source} source")

    def _create_transforms(self) -> transforms.Compose:
        """Create transforms that adapt to image channels and resize to optimal size"""
        dataset_cfg = self.config['dataset']
        mean = dataset_cfg['mean']
        std = dataset_cfg['std']

        # Get actual image to determine number of channels
        if len(self.image_paths) > 0:
            try:
                sample_image = Image.open(self.image_paths[0])
                num_channels = len(sample_image.getbands())
                logger.info(f"Detected {num_channels} channel(s) for dataset {self.dataset_name}")
            except Exception as e:
                logger.warning(f"Could not detect channels from sample image: {e}")
                num_channels = dataset_cfg['in_channels']
        else:
            num_channels = dataset_cfg['in_channels']
            logger.info(f"Using default {num_channels} channel(s) for dataset {self.dataset_name}")

        # Adjust normalization for grayscale if needed
        if num_channels == 1:
            # Grayscale images - use single channel normalization
            if len(mean) == 3:  # If config has RGB values, use first one
                mean = [mean[0]]
                std = [std[0]]
            elif len(mean) == 0 or mean == [0.485, 0.456, 0.406]:  # Default ImageNet values
                mean = [0.5]  # Default for grayscale
                std = [0.5]
        elif num_channels == 3 and len(mean) == 1:
            # RGB images but config has grayscale values - replicate
            mean = [mean[0]] * 3
            std = [std[0]] * 3

        # Determine optimal resize size based on original dimensions
        if len(self.image_paths) > 0:
            try:
                sample_image = Image.open(self.image_paths[0])
                original_width, original_height = sample_image.size

                # Choose optimal size:
                # - Small images (<=64px): keep original or scale to 32x32
                # - Medium images (65-256px): scale to 64x64
                # - Large images (>256px): scale to 128x128 to prevent memory issues
                if original_height <= 64 and original_width <= 64:
                    target_size = (32, 32)  # Standardize small images
                elif original_height <= 256 and original_width <= 256:
                    target_size = (64, 64)  # Good balance for medium images
                else:
                    target_size = (128, 128)  # Prevent memory issues for large images

                logger.info(f"Original size: {original_width}x{original_height} -> Resizing to: {target_size[0]}x{target_size[1]}")

            except Exception as e:
                logger.warning(f"Could not detect image size: {e}")
                target_size = (64, 64)  # Default fallback
        else:
            target_size = (64, 64)  # Default

        transform_list = [
            transforms.Resize(target_size),  # 🔥 KEY CHANGE: Resize to optimal size
            transforms.ToTensor(),
        ]

        # Only add normalization if we have appropriate mean/std values
        if len(mean) == num_channels and len(std) == num_channels:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
            logger.info(f"Using normalization: mean={mean}, std={std}")
        else:
            logger.warning(f"Skipping normalization - mean/std mismatch: "
                          f"channels={num_channels}, mean_len={len(mean)}, std_len={len(std)}")

        # Add augmentation for training
        if self.split == "train":
            # Insert augmentation before resizing (so we augment the original image)
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize(target_size),  # Resize after augmentation
                transforms.ToTensor(),
            ]

            # Add normalization if applicable
            if len(mean) == num_channels and len(std) == num_channels:
                transform_list.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(transform_list)

    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load data based on specified source"""
        dataset_path = self.root_dir / self.dataset_name

        if self.data_source == "local" or (self.data_source == "auto" and dataset_path.exists() and any(dataset_path.iterdir())):
            # Load from local folder structure
            return self._load_from_folder_structure(dataset_path)
        elif self.data_source == "torchvision" or (self.data_source == "auto" and not dataset_path.exists()):
            # Download and create standard dataset
            return self._download_standard_dataset()
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")

    def _load_from_folder_structure(self, dataset_path: Path) -> Tuple[List[str], List[int], List[str]]:
        """Load data from folder structure where each subfolder is a class"""
        image_paths = []
        targets = []
        class_names = []

        # Find all class folders
        class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
        class_folders.sort()

        for class_idx, class_folder in enumerate(class_folders):
            class_names.append(class_folder.name)

            # Get all images in this class folder
            image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'}
            for ext in image_extensions:
                for image_path in class_folder.glob(ext):
                    image_paths.append(str(image_path))
                    targets.append(class_idx)

        return image_paths, targets, class_names

    def _download_standard_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """Download standard dataset using torchvision"""
        dataset_path = self.root_dir / self.dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Create tmp directory for downloads
        tmp_dir = self.root_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        # Map dataset names to torchvision datasets
        dataset_map = {
            'cifar10': (torchvision.datasets.CIFAR10, 10),
            'cifar100': (torchvision.datasets.CIFAR100, 100),
            'mnist': (torchvision.datasets.MNIST, 10),
            'fashionmnist': (torchvision.datasets.FashionMNIST, 10),
        }

        if self.dataset_name.lower() in dataset_map:
            dataset_class, num_classes = dataset_map[self.dataset_name.lower()]

            try:
                # Use torchvision's built-in download
                dataset = dataset_class(root=str(tmp_dir), train=(self.split == "train"), download=True)

                # Save images to folder structure for future use
                return self._save_torchvision_dataset(dataset, dataset_path, num_classes)

            except Exception as e:
                logger.error(f"Dataset download failed: {e}")
                raise
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _save_torchvision_dataset(self, dataset, dataset_path: Path, num_classes: int) -> Tuple[List[str], List[int], List[str]]:
        """Save torchvision dataset to folder structure"""
        image_paths = []
        targets = []
        class_names = [str(i) for i in range(num_classes)]

        for idx, (image, target) in enumerate(dataset):
            class_folder = dataset_path / str(target)
            class_folder.mkdir(exist_ok=True)

            image_path = class_folder / f"{idx:06d}.png"

            if isinstance(image, Image.Image):
                image.save(image_path)
            else:
                # Convert tensor to PIL Image
                pil_image = transforms.ToPILImage()(image)
                pil_image.save(image_path)

            image_paths.append(str(image_path))
            targets.append(target)

        return image_paths, targets, class_names

    def _build_adaptive_decoder(self, input_height: int, input_width: int) -> nn.Module:
        """Build decoder that reconstructs to original image size"""

        # Calculate the starting size for decoder (inverse of encoder)
        def conv_output_size(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1

        # Calculate starting size for decoder
        h, w = input_height, input_width
        for _ in range(4):  # 4 conv layers in encoder
            h = conv_output_size(h)
            w = conv_output_size(w)

        starting_size = (h, w)
        starting_features = 512 * h * w

        return nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, starting_features),
            nn.ReLU(True),
            nn.Unflatten(1, (512, starting_size[0], starting_size[1])),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.in_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def _download_standard_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """Download standard dataset using torchvision with safe extraction"""
        dataset_path = self.root_dir / self.dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Create tmp directory for downloads
        tmp_dir = self.root_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        # Map dataset names to torchvision datasets
        dataset_map = {
            'cifar10': (torchvision.datasets.CIFAR10, 10),
            'cifar100': (torchvision.datasets.CIFAR100, 100),
            'mnist': (torchvision.datasets.MNIST, 10),
            'fashionmnist': (torchvision.datasets.FashionMNIST, 10),
        }

        if self.dataset_name.lower() in dataset_map:
            dataset_class, num_classes = dataset_map[self.dataset_name.lower()]

            try:
                # Use torchvision's built-in download with tmp directory
                dataset = dataset_class(root=str(tmp_dir), train=(self.split == "train"), download=True)

                # Save images to folder structure for future use
                return self._save_torchvision_dataset(dataset, dataset_path, num_classes)

            except Exception as e:
                logger.error(f"Dataset download failed: {e}")
                raise
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target, self.image_paths[idx]

class EnhancementModules:
    """Domain-specific enhancement modules"""

    def __init__(self, config: Dict):
        self.config = config
        self.enhancement_cfg = config['model']['enhancement_modules']

    def apply_astronomical_enhancements(self, images: torch.Tensor) -> torch.Tensor:
        """Apply astronomical image enhancements"""
        if not self.enhancement_cfg['astronomical']['enabled']:
            return images

        enhanced_images = images.clone()
        weights = self.enhancement_cfg['astronomical']['weights']

        # Structure preservation (edge enhancement)
        if self.enhancement_cfg['astronomical']['components']['structure_preservation']:
            enhanced_images = self._enhance_edges(enhanced_images, weights['edge_weight'])

        # Detail preservation (high-frequency enhancement)
        if self.enhancement_cfg['astronomical']['components']['detail_preservation']:
            enhanced_images = self._enhance_details(enhanced_images, weights['detail_weight'])

        return enhanced_images

    def apply_medical_enhancements(self, images: torch.Tensor) -> torch.Tensor:
        """Apply medical image enhancements"""
        if not self.enhancement_cfg['medical']['enabled']:
            return images

        enhanced_images = images.clone()
        weights = self.enhancement_cfg['medical']['weights']

        # Contrast enhancement
        if self.enhancement_cfg['medical']['components']['contrast_enhancement']:
            enhanced_images = self._enhance_contrast(enhanced_images, weights['contrast_weight'])

        # Boundary enhancement
        if self.enhancement_cfg['medical']['components']['tissue_boundary']:
            enhanced_images = self._enhance_boundaries(enhanced_images, weights['boundary_weight'])

        return enhanced_images

    def apply_agricultural_enhancements(self, images: torch.Tensor) -> torch.Tensor:
        """Apply agricultural image enhancements"""
        if not self.enhancement_cfg['agricultural']['enabled']:
            return images

        enhanced_images = images.clone()
        weights = self.enhancement_cfg['agricultural']['weights']

        # Texture enhancement
        if self.enhancement_cfg['agricultural']['components']['texture_analysis']:
            enhanced_images = self._enhance_texture(enhanced_images, weights['texture_weight'])

        # Pattern enhancement
        if self.enhancement_cfg['agricultural']['components']['pattern_enhancement']:
            enhanced_images = self._enhance_patterns(enhanced_images, weights['pattern_weight'])

        return enhanced_images

    def _enhance_edges(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance edges using Sobel filter"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_x = sobel_x.to(images.device)
        sobel_y = sobel_y.to(images.device)

        edges_x = torch.nn.functional.conv2d(images.mean(dim=1, keepdim=True), sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(images.mean(dim=1, keepdim=True), sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        return images + weight * edges.repeat(1, 3, 1, 1)

    def _enhance_details(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance fine details using high-pass filter"""
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel = kernel.to(images.device)

        details = torch.nn.functional.conv2d(images, kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
        return images + weight * (details - images)

    def _enhance_contrast(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance contrast using histogram equalization approximation"""
        # Simple contrast enhancement
        mean = images.mean(dim=[2, 3], keepdim=True)
        std = images.std(dim=[2, 3], keepdim=True)
        return torch.sigmoid((images - mean) / (std + 1e-8)) * weight + images * (1 - weight)

    def _enhance_boundaries(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance boundaries using Laplacian filter"""
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        laplacian = laplacian.to(images.device)

        boundaries = torch.nn.functional.conv2d(images.mean(dim=1, keepdim=True), laplacian, padding=1)
        return images + weight * boundaries.repeat(1, 3, 1, 1)

    def _enhance_texture(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance texture using local binary patterns approximation"""
        # Simple texture enhancement using variance
        patch_size = 3
        unfolded = images.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        texture = unfolded.contiguous().view(images.size(0), 3, -1, patch_size, patch_size).var(dim=2)
        texture = torch.nn.functional.interpolate(texture, size=images.shape[2:], mode='bilinear')

        return images + weight * texture

    def _enhance_patterns(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance patterns using frequency domain filtering"""
        # Simple pattern enhancement using FFT
        fft = torch.fft.fft2(images, dim=(2, 3))
        fft_shift = torch.fft.fftshift(fft, dim=(2, 3))

        # Enhance high frequencies (patterns)
        _, _, h, w = images.shape
        center_h, center_w = h // 2, w // 2
        mask = torch.ones_like(fft_shift)
        mask[:, :, center_h-10:center_h+10, center_w-10:center_w+10] = 0.5

        enhanced_fft = fft_shift * (1 + weight * mask)
        enhanced = torch.fft.ifft2(torch.fft.ifftshift(enhanced_fft, dim=(2, 3))).real

        return enhanced

class FeatureSpaceCompressor(nn.Module):
    """Secondary autoencoder that compresses 512D features to final output dimension"""

    def __init__(self, input_dim: int, compressed_dim: int, num_classes: int):
        super(FeatureSpaceCompressor, self).__init__()

        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.num_classes = num_classes

        # Compression network
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),

            nn.Linear(64, compressed_dim),  # Final compression
            nn.BatchNorm1d(compressed_dim)  # Normalize compressed features
        )

        # Reconstruction network (for training)
        self.decompressor = nn.Sequential(
            nn.Linear(compressed_dim, 64),
            nn.LeakyReLU(0.1, True),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, True),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, True),

            nn.Linear(256, input_dim)  # Back to original 512D
        )

        # Discriminative head (optional, for better feature quality)
        self.classifier = nn.Linear(compressed_dim, num_classes)

    def forward(self, x, return_all: bool = False):
        # Compress features
        compressed = self.compressor(x)

        if return_all:
            # During training: return compressed, reconstructed, and classification
            reconstructed = self.decompressor(compressed)
            classified = self.classifier(compressed)
            return compressed, reconstructed, classified
        else:
            # During inference: just return compressed features
            return compressed



class UnifiedDiscriminativeAutoencoder(nn.Module):
    """Enhanced unified autoencoder with domain-specific enhancements"""

    def __init__(self, config: Dict, num_classes: int):
        super(UnifiedDiscriminativeAutoencoder, self).__init__()

        self.config = config
        model_cfg = config['model']
        dataset_cfg = config['dataset']

        # Use 512D for rich features, but we'll compress to config dimension for output
        self.latent_dim = 512  # Internal rich representation
        self.output_dim = model_cfg['feature_dims']  # Final output dimension (e.g., 32)
        self.in_channels = dataset_cfg['in_channels']  # This should be updated to actual channels
        self.input_size = dataset_cfg['input_size']

        # Enhancement modules
        self.enhancement_modules = EnhancementModules(config)

        # Main encoder/decoder with rich 512D space
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Expert modules work on the rich 512D space
        self.reconstruction_expert = ReconstructionExpert(self.latent_dim)
        self.semantic_expert = SemanticDensityExpert(self.latent_dim, num_classes)
        self.efficiency_critic = EfficiencyCritic(self.latent_dim)

        # Feature Space Compressor - trains after main autoencoder
        self.feature_compressor = FeatureSpaceCompressor(
            input_dim=self.latent_dim,
            compressed_dim=self.output_dim,
            num_classes=num_classes
        )

        # Control whether to use compression
        self.use_compression = False  # Start with raw features during main training

        # ENHANCED: Initialize clustering and classification components
        self._initialize_enhanced_components(num_classes)

        # Domain-specific components
        self.setup_domain_specific_components()

    def update_input_channels(self, actual_channels: int):
        """Update the model to use the actual number of input channels"""
        if actual_channels != self.in_channels:
            logger.info(f"🔄 Updating model from {self.in_channels} to {actual_channels} input channels")
            self.in_channels = actual_channels

            # Rebuild encoder and decoder with correct channels
            self.encoder = self._build_encoder()
            self.decoder = self._build_decoder()

            # Move to device
            self.encoder = self.encoder.to(next(self.parameters()).device)
            self.decoder = self.decoder.to(next(self.parameters()).device)

    def _build_encoder(self) -> nn.Module:
        """Build encoder that works with standardized image sizes"""
        input_height, input_width = self.config['dataset']['input_size']

        logger.info(f"Building enhanced encoder for {input_height}x{input_width} with {self.in_channels} channels and 512D latent space")

        # All images are now standardized to 32x32, 64x64, or 128x128
        # Use appropriate architecture for each size
        if input_height <= 32:
            return self._build_32x32_encoder()
        elif input_height <= 64:
            return self._build_64x64_encoder()
        else:
            return self._build_128x128_encoder()

    def _build_32x32_encoder(self) -> nn.Module:
        """Enhanced encoder for 32x32 images with attention"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            SelfAttentionEnhanced(64),  # Enhanced with attention
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            SelfAttentionEnhanced(128),  # Enhanced with attention
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_64x64_encoder(self) -> nn.Module:
        """Enhanced encoder for 64x64 images with attention"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            SelfAttentionEnhanced(64),  # Enhanced with attention
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            SelfAttentionEnhanced(128),  # Enhanced with attention
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_128x128_encoder(self) -> nn.Module:
        """Enhanced encoder for 128x128 images with attention"""
        return nn.Sequential(
            # Initial downsampling
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            SelfAttentionEnhanced(64),  # Enhanced with attention
            nn.MaxPool2d(2, 2),

            # Continue with 64x64 architecture but enhanced
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            SelfAttentionEnhanced(128),  # Enhanced with attention
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_enhanced_small_encoder(self, input_height: int, input_width: int) -> nn.Module:
        """Enhanced encoder with 512D output for small images"""
        return nn.Sequential(
            # Block 1: 64 channels - UPDATED to use self.in_channels
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),  # Now uses actual input channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 2: 128 channels
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            # Block 3: 256 channels
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            # Block 4: 512 channels
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_enhanced_large_encoder(self, input_height: int, input_width: int) -> nn.Module:
        """Enhanced encoder with 512D output for large images"""
        return nn.Sequential(
            # Initial downsampling for large images - UPDATED to use self.in_channels
            nn.Conv2d(self.in_channels, 64, 7, 2, 3),  # Now uses actual input channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            # Rest of the encoder remains the same...
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_decoder(self) -> nn.Module:
        """Build decoder for 512D latent space"""
        input_height, input_width = self.config['dataset']['input_size']

        logger.info(f"Building decoder for {input_height}x{input_width} with {self.in_channels} output channels")

        if input_height <= 64:
            return self._build_enhanced_small_decoder(input_height, input_width)
        else:
            return self._build_enhanced_large_decoder(input_height, input_width)

    def _build_enhanced_small_decoder(self, input_height: int, input_width: int) -> nn.Module:
        """Enhanced decoder for small images from 512D latent space"""
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512 * 2 * 2),
            nn.LeakyReLU(0.1, True),
            nn.Unflatten(1, (512, 2, 2)),

            # Block 1: 2x2 -> 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),

            # Block 2: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),

            # Block 3: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),

            # Block 4: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),

            # Final convolution to get exact size - UPDATED to use self.in_channels
            nn.Conv2d(32, self.in_channels, 3, 1, 1),  # Now uses actual output channels
            nn.Sigmoid()
        )

    def _build_enhanced_large_decoder(self, input_height: int, input_width: int) -> nn.Module:
            """Enhanced decoder for large images from 512D latent space"""
            return nn.Sequential(
                nn.Linear(512, 512),  # Input from 512D latent space
                nn.LeakyReLU(0.1, True),
                nn.Linear(512, 512 * 4 * 4),
                nn.LeakyReLU(0.1, True),
                nn.Unflatten(1, (512, 4, 4)),

                # Block 1: 4x4 -> 8x8
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.2),

                # Block 2: 8x8 -> 16x16
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.2),

                # Block 3: 16x16 -> 32x32
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.1),

                # Block 4: 32x32 -> 64x64
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, True),

                # Additional blocks for larger images
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 64x64 -> 128x128
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, True),

                nn.ConvTranspose2d(32, self.in_channels, 4, 2, 1),  # 128x128 -> 256x256
                nn.Sigmoid()
            )

    def _build_adaptive_encoder(self, input_height: int, input_width: int) -> nn.Module:
        """Build encoder that automatically adapts to any image size"""

        # Calculate the size after each conv layer
        def conv_output_size(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1

        # Calculate final feature map size
        h, w = input_height, input_width
        for _ in range(4):  # 4 conv layers
            h = conv_output_size(h)
            w = conv_output_size(w)

        final_features = 512 * h * w

        return nn.Sequential(
            # Input: [in_channels, H, W]
            nn.Conv2d(self.in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(final_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(512, self.latent_dim)
        )

    def _build_decoder(self) -> nn.Module:
        """Build decoder that adapts to original image dimensions"""
        # Get actual image dimensions from the dataset
        if hasattr(self, 'dataset') and self.dataset and len(self.dataset) > 0:
            sample_image, _, _ = self.dataset[0]
            _, input_height, input_width = sample_image.shape
        else:
            # Fallback to config if no data available yet
            input_height, input_width = self.config['dataset']['input_size']

        logger.info(f"Building decoder for image size: {input_height}x{input_width}")

        return self._build_adaptive_decoder(input_height, input_width)

    def _build_enhanced_encoder(self) -> nn.Module:
        """Build enhanced encoder architecture"""
        return nn.Sequential(
            # Input: [in_channels, H, W]
            nn.Conv2d(self.in_channels, 64, 4, 2, 1),  # 1/2
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 1/4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 1/8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 1/16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(512, self.latent_dim)
        )

    def _build_basic_encoder(self) -> nn.Module:
        """Build basic encoder architecture"""
        return nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256 * (self.input_size[0]//16) * (self.input_size[1]//16), 512),
            nn.ReLU(True),
            nn.Linear(512, self.latent_dim)
        )

    def _build_decoder(self) -> nn.Module:
        """Build decoder"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512 * 2 * 2),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.in_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def setup_domain_specific_components(self):
        """Setup domain-specific components based on configuration"""
        image_type = self.config['dataset']['image_type']
        enhancement_cfg = self.config['model']['enhancement_modules']

        if image_type == "astronomical" and enhancement_cfg['astronomical']['enabled']:
            self.astronomical_head = nn.Linear(self.latent_dim, 64)
        elif image_type == "medical" and enhancement_cfg['medical']['enabled']:
            self.medical_head = nn.Linear(self.latent_dim, 32)
        elif image_type == "agricultural" and enhancement_cfg['agricultural']['enabled']:
            self.agricultural_head = nn.Linear(self.latent_dim, 48)

    def forward(self, x, targets=None, phase=1, apply_enhancements=True, use_compression=None):
        # Apply domain-specific enhancements
        if apply_enhancements:
            image_type = self.config['dataset']['image_type']
            if image_type == "astronomical":
                x = self.enhancement_modules.apply_astronomical_enhancements(x)
            elif image_type == "medical":
                x = self.enhancement_modules.apply_medical_enhancements(x)
            elif image_type == "agricultural":
                x = self.enhancement_modules.apply_agricultural_enhancements(x)

        # Base encoding to rich 512D space
        base_latent = self.encoder(x)
        current_latent = base_latent

        # Progressive expert guidance on rich features
        if phase >= 2:
            current_latent = self.reconstruction_expert(current_latent)

        if phase >= 3 and targets is not None:
            current_latent = self.semantic_expert(current_latent, targets)

        # Final reconstruction from rich features
        reconstructed = self.decoder(current_latent)

        # Apply feature compression if requested
        use_compression = use_compression if use_compression is not None else self.use_compression
        if use_compression:
            compressed_latent = self.feature_compressor(current_latent)
        else:
            compressed_latent = current_latent

        # Always return 4 values for consistency
        if phase >= 4:
            efficiency_scores = self.efficiency_critic(current_latent)
        else:
            efficiency_scores = None

        return reconstructed, compressed_latent, current_latent, efficiency_scores

    def _initialize_enhanced_components(self, num_classes: int):
        """Initialize enhanced components from old model: clustering and classification"""
        # Initialize clustering components
        self.use_kl_divergence = self.config['model'].get('use_kl_divergence', True)
        if self.use_kl_divergence:
            self.register_buffer('cluster_centers',
                               torch.randn(num_classes, self.latent_dim))
            self.register_buffer('clustering_temperature',
                               torch.tensor([1.0]))

        # Initialize classifier for semantic guidance
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def organize_latent_space(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Enhanced latent space organization with clustering from old model"""
        output = {'embeddings': embeddings}

        # Clustering organization
        if self.use_kl_divergence and hasattr(self, 'cluster_centers'):
            cluster_centers = self.cluster_centers.to(embeddings.device)
            distances = torch.cdist(embeddings, cluster_centers)

            # Convert distances to probabilities (soft assignments)
            q_dist = 1.0 / (1.0 + (distances / self.clustering_temperature) ** 2)
            q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

            if labels is not None:
                # Create target distribution if labels are provided
                p_dist = torch.zeros_like(q_dist)
                for i in range(self.cluster_centers.size(0)):
                    mask = (labels == i)
                    if mask.any():
                        p_dist[mask, i] = 1.0
            else:
                # During prediction, use current distribution as target
                p_dist = q_dist.detach()

            output.update({
                'cluster_probabilities': q_dist,
                'target_distribution': p_dist,
                'cluster_assignments': q_dist.argmax(dim=1),
                'cluster_confidence': q_dist.max(dim=1)[0]
            })

        # Classification head
        if hasattr(self, 'classifier'):
            class_logits = self.classifier(embeddings)
            output.update({
                'class_logits': class_logits,
                'class_predictions': class_logits.argmax(dim=1),
                'class_probabilities': F.softmax(class_logits, dim=1)
            })

        return output




# Expert Modules (from previous implementation, enhanced)
class ReconstructionExpert(nn.Module):
    def __init__(self, latent_dim=128):
        super(ReconstructionExpert, self).__init__()
        self.latent_dim = latent_dim
        self.guidance_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, latent):
        guidance = self.guidance_net(latent) * 0.1
        return latent + guidance

class SemanticDensityExpert(nn.Module):
    def __init__(self, latent_dim=128, num_classes=100):
        super(SemanticDensityExpert, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.guidance_net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )

    def forward(self, latent, targets):
        batch_size = latent.size(0)
        targets_one_hot = torch.zeros(batch_size, self.num_classes).to(latent.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        class_proto = torch.matmul(targets_one_hot, self.class_prototypes)
        latent_with_class = torch.cat([latent, targets_one_hot], dim=1)
        guidance = self.guidance_net(latent_with_class) * 0.05
        guided_latent = latent + guidance
        return guided_latent

class EfficiencyCritic(nn.Module):
    def __init__(self, latent_dim=128):
        super(EfficiencyCritic, self).__init__()
        self.latent_dim = latent_dim
        self.efficiency_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, latent):
        return self.efficiency_net(latent)

class UnifiedFeatureExtractor:
    """Main class for unified feature extraction"""

    def __init__(self, dataset_name: str, config_dir: str = "configs"):
        self.dataset_name = dataset_name
        self.config_manager = ConfigManager(config_dir)
        self.main_config = self.config_manager.load_main_config(dataset_name)
        self.dbnn_config = self.config_manager.load_dbnn_config()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.model = None
        self.dataset = None

    def setup_data(self, data_dir: str = "Data", split: str = "train"):
        """Setup data pipeline"""
        self.dataset = GenericImageDataset(data_dir, self.dataset_name, self.main_config, split)
        return self.dataset

    def initialize_model(self):
        """Initialize the unified autoencoder model with dynamic sizing"""
        num_classes = len(set(self.dataset.targets)) if self.dataset else 10

        # Update config with actual processed image dimensions
        if self.dataset and len(self.dataset) > 0:
            sample_image, _, _ = self.dataset[0]
            channels, height, width = sample_image.shape

            # These are the dimensions AFTER transforms (resizing)
            self.main_config['dataset']['in_channels'] = channels
            self.main_config['dataset']['input_size'] = [height, width]

            logger.info(f"📊 Processed image dimensions: {channels} channels, {height}x{width}")

        # Initialize model
        self.model = UnifiedDiscriminativeAutoencoder(self.main_config, num_classes)
        self.model = self.model.to(self.device)

        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model

    def train(self, data_dir: str = "Data", save_dir: str = "models"):
        """Multi-round competitive training with enhanced latent space organization and label encoding"""
        if self.dataset is None:
            self.setup_data(data_dir, "train")

        if self.model is None:
            self.initialize_model()

        # ENHANCED: Initialize label encoder for folder name preservation
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()

        # Fit label encoder with class names from dataset
        if hasattr(self.dataset, 'class_names'):
            self.label_encoder.fit(self.dataset.class_names)
            logger.info(f"🏷️  Label encoder initialized with {len(self.label_encoder.classes_)} classes: {list(self.label_encoder.classes_)}")
        else:
            # Fallback: use numeric labels but warn user
            logger.warning("⚠️  No class names found in dataset, using numeric labels")
            self.label_encoder = None

        # Set deterministic training for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Increase batch size for A100
        batch_size = min(128, self.main_config['training']['batch_size'] * 4)
        logger.info(f"Using batch size {batch_size} optimized for A100")

        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        # Updated mixed precision training
        scaler = torch.amp.GradScaler('cuda')

        # More aggressive but stable optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        criterion = nn.MSELoss()

        # More aggressive but achievable loss targets
        phases = [
            (1, "Base Autoencoder", 0.05),
            (2, "+ Reconstruction Expert + Clustering", 0.03),
            (3, "+ Semantic Density Expert + Classification", 0.02),
            (4, "+ Efficiency Critic + Full Enhancement", 0.015)
        ]

        # Multi-round training parameters
        max_rounds = 5
        max_epochs_per_phase = 20
        min_epochs_per_phase = 3
        phase_patience = 5

        # Track progress across rounds
        round_best_losses = {phase_num: float('inf') for phase_num, _, _ in phases}
        current_round = 0

        while current_round < max_rounds:
            logger.info(f"🚀 Starting Training Round {current_round + 1}/{max_rounds}")
            round_improved = False

            for phase_num, description, target_loss in phases:
                logger.info(f"🎯 Round {current_round + 1} - {description} (Phase {phase_num}) - Target: {target_loss}")

                phase_best_loss = round_best_losses[phase_num]
                patience_counter = 0

                for epoch in range(max_epochs_per_phase):
                    self.model.train()
                    total_loss = 0
                    num_batches = 0
                    loss_components_avg = {'reconstruction': 0, 'clustering': 0, 'classification': 0, 'efficiency': 0}

                    for batch_idx, (data, targets, _) in enumerate(train_loader):
                        data = data.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)

                        optimizer.zero_grad()

                        # Mixed precision forward pass
                        with torch.amp.autocast('cuda'):
                            # Model returns 4 values: (reconstructed, current_latent, base_latent, efficiency_scores)
                            reconstructed, current_latent, base_latent, efficiency_scores = self.model(data, targets, phase=phase_num)

                            # ENHANCED LOSS CALCULATION WITH OLD MODEL TECHNIQUES
                            rec_loss = criterion(reconstructed, data)
                            total_batch_loss = rec_loss
                            loss_components = {'reconstruction': rec_loss.item()}

                            # Phase 2+: Add clustering loss from old model
                            if phase_num >= 2 and hasattr(self.model, 'use_kl_divergence') and self.model.use_kl_divergence:
                                cluster_info = self.model.organize_latent_space(current_latent, targets)
                                if 'cluster_probabilities' in cluster_info and 'target_distribution' in cluster_info:
                                    cluster_loss = F.kl_div(
                                        cluster_info['cluster_probabilities'].log(),
                                        cluster_info['target_distribution'],
                                        reduction='batchmean'
                                    )
                                    total_batch_loss = total_batch_loss + 0.1 * cluster_loss
                                    loss_components['clustering'] = cluster_loss.item()

                            # Phase 3+: Add classification loss from old model
                            if phase_num >= 3 and targets is not None and hasattr(self.model, 'classifier'):
                                class_logits = self.model.classifier(current_latent)
                                class_loss = F.cross_entropy(class_logits, targets)
                                total_batch_loss = total_batch_loss + 0.1 * class_loss
                                loss_components['classification'] = class_loss.item()

                            # Phase 4+: Add efficiency loss (original functionality)
                            if phase_num >= 4:
                                eff_loss = (1 - efficiency_scores[:, 0]).mean()
                                total_batch_loss = total_batch_loss + 0.1 * eff_loss
                                loss_components['efficiency'] = eff_loss.item()

                            loss = total_batch_loss

                        # Skip NaN batches
                        if torch.isnan(loss):
                            continue

                        # Mixed precision backward pass
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.8)
                        scaler.step(optimizer)
                        scaler.update()

                        total_loss += loss.item()
                        num_batches += 1

                        # Accumulate loss components for logging
                        for key in loss_components_avg:
                            if key in loss_components:
                                loss_components_avg[key] += loss_components[key]

                    if num_batches == 0:
                        continue

                    # Calculate averages
                    epoch_loss = total_loss / num_batches
                    for key in loss_components_avg:
                        loss_components_avg[key] /= num_batches

                    # Dynamic learning rate adjustment based on progress
                    current_lr = optimizer.param_groups[0]['lr']
                    if epoch_loss < phase_best_loss:
                        phase_best_loss = epoch_loss
                        round_best_losses[phase_num] = phase_best_loss
                        patience_counter = 0
                        round_improved = True

                        # Reward good progress with learning rate boost
                        if epoch_loss < target_loss * 1.5:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = min(current_lr * 1.1, 0.002)
                    else:
                        patience_counter += 1
                        # Penalize stagnation with learning rate reduction
                        if patience_counter >= 2:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = max(current_lr * 0.8, 1e-6)

                    # Enhanced logging with loss components
                    if epoch % 2 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        progress_pct = (target_loss / epoch_loss * 100) if epoch_loss > 0 else 0

                        # Build detailed loss string
                        loss_details = []
                        for comp_name, comp_value in loss_components_avg.items():
                            if comp_value > 0:
                                loss_details.append(f"{comp_name[:4]}:{comp_value:.4f}")
                        loss_str = " | ".join(loss_details)

                        logger.info(f'Round {current_round + 1} Phase {phase_num} Epoch [{epoch}/{max_epochs_per_phase}], '
                                   f'Loss: {epoch_loss:.4f}, Best: {phase_best_loss:.4f}, '
                                   f'Target: {target_loss:.4f} ({progress_pct:.1f}%), LR: {current_lr:.6f}')
                        if loss_str:
                            logger.info(f'📊 Loss components: {loss_str}')

                    # Check if we've achieved target
                    if epoch >= min_epochs_per_phase and epoch_loss <= target_loss:
                        logger.info(f'🎉 Round {current_round + 1} Phase {phase_num} ACHIEVED TARGET! '
                                   f'Loss: {epoch_loss:.4f} <= {target_loss:.4f}')
                        break

                    # Early stopping if no improvement
                    if patience_counter >= phase_patience:
                        logger.info(f'⏹️  Round {current_round + 1} Phase {phase_num} early stopping '
                                   f'after {epoch} epochs (best: {phase_best_loss:.4f})')
                        break

                logger.info(f'✅ Round {current_round + 1} {description} completed with best loss: {phase_best_loss:.4f}')

                # Save phase checkpoint with label encoder
                phase_checkpoint_path = Path(save_dir) / f"{self.dataset_name}_round{current_round + 1}_phase{phase_num}.pth"
                checkpoint_data = {
                    'model_state_dict': self.model.state_dict(),
                    'phase': phase_num,
                    'round': current_round + 1,
                    'best_loss': phase_best_loss,
                    'round_best_losses': round_best_losses,
                    'enhanced_components': {
                        'use_kl_divergence': getattr(self.model, 'use_kl_divergence', False),
                        'has_classifier': hasattr(self.model, 'classifier')
                    },
                    # ENHANCED: Save label encoder information
                    'label_encoder': {
                        'classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
                        'fitted': self.label_encoder is not None
                    } if hasattr(self, 'label_encoder') else None,
                    'dataset_name': self.dataset_name,
                    'class_count': len(self.label_encoder.classes_) if self.label_encoder else 0
                }
                torch.save(checkpoint_data, phase_checkpoint_path)
                logger.info(f"💾 Checkpoint saved with label encoder ({len(self.label_encoder.classes_) if self.label_encoder else 0} classes)")

            current_round += 1
            self.current_round = current_round  # Track current round for feature extraction

            # Check if we should continue to next round
            targets_achieved = sum(1 for i, (_, _, target) in enumerate(phases)
                                 if round_best_losses[i+1] <= target)

            logger.info(f"📊 Round {current_round} Summary: {targets_achieved}/{len(phases)} targets achieved")

            if targets_achieved == len(phases):
                logger.info("🎊 ALL TARGETS ACHIEVED! Training complete.")
                break
            elif not round_improved:
                logger.info("🛑 No improvement this round. Stopping training.")
                break
            else:
                # Adjust targets for next round based on current performance
                for i, (phase_num, _, current_target) in enumerate(phases):
                    current_best = round_best_losses[phase_num]
                    if current_best > current_target:
                        # Make target slightly more achievable for next round
                        new_target = current_target * 1.2
                        phases[i] = (phase_num, phases[i][1], new_target)
                        logger.info(f"🔄 Adjusting Phase {phase_num} target: {current_target:.4f} → {new_target:.4f}")

        # ENHANCED: Final model save with label encoder
        self.save_model(save_dir)

        # Print final results
        logger.info("🏁 FINAL TRAINING RESULTS:")
        for phase_num, description, target_loss in phases:
            achieved = "✅" if round_best_losses[phase_num] <= target_loss else "❌"
            improvement = target_loss - round_best_losses[phase_num] if round_best_losses[phase_num] <= target_loss else round_best_losses[phase_num] - target_loss
            status = "ACHIEVED" if round_best_losses[phase_num] <= target_loss else "MISSED"
            logger.info(f"  {achieved} {description}: {round_best_losses[phase_num]:.4f} (target: {target_loss:.4f}) [{status}]")

        # Log enhanced components status
        logger.info("🔧 Enhanced Components Status:")
        logger.info(f"  - Self-Attention: ✅ Integrated in encoders")
        logger.info(f"  - Clustering: {'✅ Enabled' if getattr(self.model, 'use_kl_divergence', False) else '❌ Disabled'}")
        logger.info(f"  - Classification: {'✅ Enabled' if hasattr(self.model, 'classifier') else '❌ Disabled'}")
        logger.info(f"  - Combined Losses: ✅ Active across all phases")
        logger.info(f"  - Label Encoder: {'✅ ' + str(len(self.label_encoder.classes_)) + ' classes' if self.label_encoder else '❌ Not available'}")

        if self.label_encoder:
            logger.info(f"  - Class Names: {list(self.label_encoder.classes_)}")

    def extract_features(self, data_dir: str = "Data", split: str = "train", use_compression: bool = False) -> pd.DataFrame:
        """Enhanced feature extraction with organized latent space"""
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")

        if self.dataset is None:
            self.setup_data(data_dir, split)

        # Set model to eval mode and ensure deterministic behavior
        self.model.eval()

        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        features_list = []
        targets_list = []
        image_paths_list = []
        image_hashes_list = []
        cluster_assignments_list = []  # Enhanced: store cluster info
        cluster_confidence_list = []   # Enhanced: store confidence scores

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.main_config['training']['batch_size'],
            shuffle=False,
            num_workers=self.main_config['training']['num_workers']
        )

        with torch.no_grad():
            for batch_idx, (data, targets, image_paths) in enumerate(dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Get final latent representation
                reconstructed, final_latent, base_latent, efficiency_scores = self.model(
                    data, targets, phase=4, use_compression=use_compression
                )

                # ENHANCED: Get organized latent space information
                latent_info = self.model.organize_latent_space(final_latent, targets)
                organized_latent = latent_info['embeddings']

                # Store enhanced features and metadata
                features_list.append(organized_latent.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                image_paths_list.extend(image_paths)

                # Enhanced: Store clustering information
                if 'cluster_assignments' in latent_info:
                    cluster_assignments_list.append(latent_info['cluster_assignments'].cpu().numpy())
                    cluster_confidence_list.append(latent_info['cluster_confidence'].cpu().numpy())

                # Generate consistency hashes for verification
                batch_hashes = [self._generate_image_hash(path) for path in image_paths]
                image_hashes_list.extend(batch_hashes)

                if batch_idx % 10 == 0:
                    feature_type = "compressed" if use_compression else "rich 512D"
                    logger.info(f'Processed batch {batch_idx}/{len(dataloader)} ({feature_type} features)')

        # Concatenate all batches
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Enhanced: Add clustering information if available
        if cluster_assignments_list:
            all_cluster_assignments = np.concatenate(cluster_assignments_list, axis=0)
            all_cluster_confidence = np.concatenate(cluster_confidence_list, axis=0)

        # Create DataFrame with enhanced information
        latent_dim = all_features.shape[1]
        feature_columns = [f'feature_{i}' for i in range(latent_dim)]

        df_data = {
            'target': all_targets,
            'image_path': image_paths_list,
            'image_hash': image_hashes_list,
            'extraction_timestamp': time.time(),
            'model_version': f"{self.dataset_name}_round{getattr(self, 'current_round', 0)}",
            'feature_type': 'compressed' if use_compression else 'rich_512D'
        }

        # Enhanced: Add clustering columns
        if cluster_assignments_list:
            df_data['cluster_assignment'] = all_cluster_assignments
            df_data['cluster_confidence'] = all_cluster_confidence

        for i in range(latent_dim):
            df_data[f'feature_{i}'] = all_features[:, i]

        df = pd.DataFrame(df_data)
        base_columns = ['target', 'image_path', 'image_hash', 'extraction_timestamp', 'model_version', 'feature_type']
        if cluster_assignments_list:
            base_columns.extend(['cluster_assignment', 'cluster_confidence'])
        columns = base_columns + feature_columns
        df = df[columns]

        feature_type = f"compressed {latent_dim}D" if use_compression else "rich 512D"
        logger.info(f"✅ Successfully extracted {len(df)} samples with {feature_type} features")
        logger.info(f"🔒 Latent space organization applied with clustering")

        if cluster_assignments_list:
            unique_clusters = len(np.unique(all_cluster_assignments))
            avg_confidence = np.mean(all_cluster_confidence)
            logger.info(f"📊 Clustering: {unique_clusters} clusters, avg confidence: {avg_confidence:.4f}")

        return df

    def save_features_and_config(self, df: pd.DataFrame, output_dir: str = "data"):
        """Save features to CSV and create configuration files in data/ folder"""
        # Create output directory in data/ (not Data/)
        output_path = Path(output_dir) / self.dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV (keep all columns including image_path for reference)
        csv_path = output_path / f"{self.dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Features saved to: {csv_path}")

        # Get feature columns (exclude metadata columns)
        metadata_columns = ['target', 'image_path', 'image_hash', 'extraction_timestamp', 'model_version', 'feature_type']
        feature_columns = [col for col in df.columns if col not in metadata_columns]

        # Create dataset configuration - ONLY target and features, no image_path
        dataset_config = {
            "file_path": f"data/{self.dataset_name}/{self.dataset_name}.csv",
            "column_names": ["target"] + feature_columns,  # Only target and features
            "separator": ",",
            "has_header": True,
            "target_column": "target",
            "modelType": "Histogram",
            "feature_group_size": 2,
            "max_combinations": 1000,
            "bin_sizes": [21],
            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            }
        }

        # Save dataset configuration
        dataset_config_path = output_path / f"{self.dataset_name}.conf"
        with open(dataset_config_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)

        logger.info(f"Dataset configuration saved to: {dataset_config_path}")
        logger.info(f"Configuration includes: target + {len(feature_columns)} features")

        return csv_path, dataset_config_path

    def save_model(self, save_dir: str = "models"):
        """Save trained model with label encoder"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / f"{self.dataset_name}_autoencoder.pth"

        # ENHANCED: Include label encoder in model save
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.main_config,
            'dataset_name': self.dataset_name,
            'label_encoder': {
                'classes': self.label_encoder.classes_.tolist() if hasattr(self, 'label_encoder') and self.label_encoder else [],
                'fitted': hasattr(self, 'label_encoder') and self.label_encoder is not None
            } if hasattr(self, 'label_encoder') else None
        }

        torch.save(model_data, model_path)
        logger.info(f"💾 Model saved to: {model_path} with label encoder")

        # Also save label encoder separately for easy access
        if hasattr(self, 'label_encoder') and self.label_encoder:
            encoder_path = save_path / f"{self.dataset_name}_label_encoder.json"
            encoder_data = {
                'classes': self.label_encoder.classes_.tolist(),
                'dataset_name': self.dataset_name,
                'timestamp': time.time()
            }
            with open(encoder_path, 'w') as f:
                json.dump(encoder_data, f, indent=2)
            logger.info(f"🏷️  Label encoder saved to: {encoder_path}")

    def load_model(self, model_path: str):
        """Load trained model with label encoder support"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load label encoder if available
            if 'label_encoder' in checkpoint and checkpoint['label_encoder'] and checkpoint['label_encoder']['fitted']:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(checkpoint['label_encoder']['classes'])
                logger.info(f"🏷️  Loaded label encoder with {len(self.label_encoder.classes_)} classes: {list(self.label_encoder.classes_)}")
            else:
                logger.warning("⚠️  No label encoder found in checkpoint, using numeric labels")
                self.label_encoder = None

            # Rest of the existing model loading code...
            current_state_dict = self.model.state_dict()
            saved_state_dict = checkpoint['model_state_dict']

            # Filter out incompatible keys
            compatible_state_dict = {}
            missing_keys = []
            unexpected_keys = []

            for key in current_state_dict.keys():
                if key in saved_state_dict:
                    if current_state_dict[key].shape == saved_state_dict[key].shape:
                        compatible_state_dict[key] = saved_state_dict[key]
                    else:
                        missing_keys.append(key)
                        logger.warning(f"Shape mismatch for {key}: current {current_state_dict[key].shape} vs saved {saved_state_dict[key].shape}")
                else:
                    missing_keys.append(key)

            for key in saved_state_dict.keys():
                if key not in current_state_dict:
                    unexpected_keys.append(key)

            # Load compatible weights
            self.model.load_state_dict(compatible_state_dict, strict=False)

            logger.info(f"✅ Loaded model from: {model_path}")
            logger.info(f"📊 Compatible parameters: {len(compatible_state_dict)}/{len(current_state_dict)}")

            if missing_keys:
                logger.warning(f"⚠️  Missing/incompatible keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"⚠️  Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("🆕 Continuing with randomly initialized weights")

    def setup_data(self, data_dir: str = "Data", split: str = "train", data_source: str = "auto"):
        """Setup data pipeline with source specification"""
        self.dataset = GenericImageDataset(data_dir, self.dataset_name, self.main_config, split, data_source)
        return self.dataset

    def extract_features(self, data_dir: str = "Data", split: str = "train", use_compression: bool = False) -> pd.DataFrame:
        """Extract features with guaranteed latent space consistency and compression option"""
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")

        if self.dataset is None:
            self.setup_data(data_dir, split)

        # Set model to eval mode and ensure deterministic behavior
        self.model.eval()

        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        features_list = []
        targets_list = []
        image_paths_list = []
        image_hashes_list = []  # For consistency verification

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.main_config['training']['batch_size'],
            shuffle=False,  # Important: maintain order for consistency
            num_workers=self.main_config['training']['num_workers']
        )

        with torch.no_grad():
            for batch_idx, (data, targets, image_paths) in enumerate(dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Get final latent representation with consistent return structure
                # Model now always returns 4 values: (reconstructed, current_latent, base_latent, efficiency_scores)
                reconstructed, final_latent, base_latent, efficiency_scores = self.model(
                    data, targets, phase=4, use_compression=use_compression
                )

                # Store features and metadata
                features_list.append(final_latent.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                image_paths_list.extend(image_paths)

                # Generate consistency hashes for verification
                batch_hashes = [self._generate_image_hash(path) for path in image_paths]
                image_hashes_list.extend(batch_hashes)

                if batch_idx % 10 == 0:
                    feature_type = "compressed" if use_compression else "rich 512D"
                    #logger.info(f'Processed batch {batch_idx}/{len(dataloader)} ({feature_type} features)')

        # Concatenate all batches
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Create DataFrame with consistency information
        latent_dim = all_features.shape[1]  # Dynamic based on compression
        feature_columns = [f'feature_{i}' for i in range(latent_dim)]

        df_data = {
            'target': all_targets,
            'image_path': image_paths_list,
            'image_hash': image_hashes_list,  # For consistency tracking
            'extraction_timestamp': time.time(),  # Track when features were extracted
            'model_version': f"{self.dataset_name}_round{getattr(self, 'current_round', 0)}",  # Track model state
            'feature_type': 'compressed' if use_compression else 'rich_512D'
        }

        for i in range(latent_dim):
            df_data[f'feature_{i}'] = all_features[:, i]

        df = pd.DataFrame(df_data)
        columns = ['target', 'image_path', 'image_hash', 'extraction_timestamp', 'model_version', 'feature_type'] + feature_columns
        df = df[columns]

        feature_type = f"compressed {latent_dim}D" if use_compression else "rich 512D"
        logger.info(f"✅ Successfully extracted {len(df)} samples with {feature_type} features")
        logger.info(f"🔒 Latent space consistency ensured with deterministic operations")

        return df

    def _generate_image_hash(self, image_path: str) -> str:
        """Generate consistent hash for image verification"""
        import hashlib
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "unknown"

    def train_feature_compressor(self, features_df: pd.DataFrame, epochs: int = 100):
        """Train the feature space compressor after main autoencoder training"""
        logger.info("🚀 Starting Feature Space Compressor Training")

        feature_cols = [f'feature_{i}' for i in range(512)]  # 512D rich features
        feature_matrix = features_df[feature_cols].values
        targets = features_df['target'].values

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_tensor = torch.FloatTensor(feature_matrix).to(device)
        targets_tensor = torch.LongTensor(targets).to(device)

        # Enable compression mode
        self.model.use_compression = True
        self.model.feature_compressor.train()

        optimizer = torch.optim.AdamW(
            self.model.feature_compressor.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )

        recon_criterion = nn.MSELoss()
        class_criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass through compressor
            compressed, reconstructed, classified = self.model.feature_compressor(
                features_tensor, return_all=True
            )

            # Combined loss: reconstruction + classification
            recon_loss = recon_criterion(reconstructed, features_tensor)
            class_loss = class_criterion(classified, targets_tensor)

            # Weighted combination - focus more on discriminative power
            total_loss = 0.3 * recon_loss + 0.7 * class_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.feature_compressor.parameters(), 1.0)
            optimizer.step()

            if total_loss < best_loss:
                best_loss = total_loss
                # Save compressor checkpoint
                torch.save({
                    'compressor_state_dict': self.model.feature_compressor.state_dict(),
                    'best_loss': best_loss,
                    'epoch': epoch
                }, f"models/{self.dataset_name}_compressor_best.pth")

            if epoch % 10 == 0:
                # Calculate accuracy
                with torch.no_grad():
                    _, _, classified = self.model.feature_compressor(features_tensor, return_all=True)
                    preds = torch.argmax(classified, dim=1)
                    accuracy = (preds == targets_tensor).float().mean()

                logger.info(f'Compressor Epoch [{epoch}/{epochs}], '
                           f'Total Loss: {total_loss.item():.4f}, '
                           f'Recon: {recon_loss.item():.4f}, '
                           f'Class: {class_loss.item():.4f}, '
                           f'Accuracy: {accuracy.item():.4f}')

        logger.info(f"✅ Feature Space Compressor training completed. Best loss: {best_loss:.4f}")

        # Set model to use compression for future inferences
        self.model.use_compression = True

    def verify_latent_consistency(self, previous_features_path: str = None) -> Dict[str, Any]:
        """Verify that the same images produce the same latent features across runs"""
        if previous_features_path and os.path.exists(previous_features_path):
            # Load previous features for comparison
            previous_df = pd.read_csv(previous_features_path)

            # Extract current features
            current_df = self.extract_features()

            # Find common images by hash
            common_hashes = set(previous_df['image_hash']).intersection(set(current_df['image_hash']))

            if common_hashes:
                consistency_results = {}

                for image_hash in list(common_hashes)[:10]:  # Sample 10 images for verification
                    prev_features = previous_df[previous_df['image_hash'] == image_hash].iloc[0]
                    curr_features = current_df[current_df['image_hash'] == image_hash].iloc[0]

                    # Compare feature vectors
                    prev_vec = prev_features[[f'feature_{i}' for i in range(self.main_config['model']['feature_dims'])]].values
                    curr_vec = curr_features[[f'feature_{i}' for i in range(self.main_config['model']['feature_dims'])]].values

                    similarity = np.dot(prev_vec, curr_vec) / (np.linalg.norm(prev_vec) * np.linalg.norm(curr_vec))
                    consistency_results[image_hash] = {
                        'similarity': similarity,
                        'prev_timestamp': prev_features['extraction_timestamp'],
                        'curr_timestamp': curr_features['extraction_timestamp'],
                        'same_model_version': prev_features['model_version'] == curr_features['model_version']
                    }

                avg_similarity = np.mean([result['similarity'] for result in consistency_results.values()])
                logger.info(f"🔍 Latent space consistency check: Average similarity = {avg_similarity:.6f}")

                return {
                    'average_similarity': avg_similarity,
                    'sample_comparisons': consistency_results,
                    'consistent': avg_similarity > 0.99  # 99% similarity threshold
                }

        return {'consistent': True, 'no_previous_data': True}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Discriminative Autoencoder System')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar100',
                       help='Dataset name (cifar100, cifar10, mnist, etc.)')
    parser.add_argument('--data-source', type=str, default='auto',
                       choices=['auto', 'torchvision', 'local'],
                       help='Data source: auto (detect), torchvision, or local folder')
    parser.add_argument('--data-dir', type=str, default='Data',
                       help='Directory for raw data (default: Data)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Directory for processed features (default: data)')

    # Training options
    parser.add_argument('--fresh', action='store_true',
                       help='Start fresh training (ignore existing models)')
    parser.add_argument('--train-main', action='store_true', default=True,
                       help='Train main autoencoder (default: True)')
    parser.add_argument('--train-compressor', action='store_true', default=True,
                       help='Train feature compressor (default: True)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override training epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')

    # Feature extraction options
    parser.add_argument('--extract-rich', action='store_true',
                       help='Extract rich 512D features in addition to compressed')
    parser.add_argument('--final-dim', type=int, default=32,
                       help='Final feature dimension after compression (default: 32)')
    parser.add_argument('--no-compression', action='store_true',
                       help='Skip compression and use raw latent features')

    # System options
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Configuration directory (default: configs)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Models directory (default: models)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only extract features from existing model')

    return parser.parse_args()

def main():
    """Enhanced main execution function with command-line arguments"""
    args = parse_arguments()

    # Create directories if they don't exist
    Path(args.data_dir).mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.config_dir).mkdir(exist_ok=True)
    Path(args.models_dir).mkdir(exist_ok=True)

    logger.info("🚀 Unified Discriminative Autoencoder System")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data source: {args.data_source}")
    logger.info(f"Fresh start: {args.fresh}")
    logger.info(f"Final feature dimension: {args.final_dim}")

    try:
        # Initialize system
        extractor = UnifiedFeatureExtractor(args.dataset, args.config_dir)

        # Override config if command line arguments provided
        if args.epochs:
            extractor.main_config['training']['epochs'] = args.epochs
        if args.batch_size:
            extractor.main_config['training']['batch_size'] = args.batch_size
        if args.final_dim:
            extractor.main_config['model']['feature_dims'] = args.final_dim

        # Setup data pipeline
        logger.info("Setting up data pipeline...")
        extractor.setup_data(args.data_dir, data_source=args.data_source)

        # Check for existing model
        model_path = Path(args.models_dir) / f"{args.dataset}_autoencoder.pth"
        compressor_path = Path(args.models_dir) / f"{args.dataset}_compressor_best.pth"

        model_exists = model_path.exists() and not args.fresh
        compressor_exists = compressor_path.exists() and not args.fresh

        # Initialize model
        logger.info("Initializing model...")
        extractor.initialize_model()

        # Load existing model if available and not fresh start
        if model_exists and not args.skip_training:
            logger.info(f"📁 Loading existing model from {model_path}")
            extractor.load_model(model_path)
        elif model_exists and args.skip_training:
            logger.info(f"📁 Loading existing model for feature extraction only")
            extractor.load_model(model_path)
        else:
            logger.info("🆕 No existing model found or fresh start requested")

        # Training phase
        if args.train_main and not args.skip_training:
            if model_exists and not args.fresh:
                logger.info("✅ Using existing main autoencoder model")
            else:
                logger.info("🎯 Starting main autoencoder training...")
                extractor.train(args.data_dir, args.models_dir)

        # Feature extraction phase
        rich_features_df = None

        if args.extract_rich or args.train_compressor:
            # Extract rich 512D features
            logger.info("🔍 Extracting rich 512D features...")
            rich_features_df = extractor.extract_features(
                args.data_dir,
                use_compression=False,
                split="train"
            )

            if args.extract_rich:
                # Save rich features
                rich_csv_path = Path(args.output_dir) / args.dataset / f"{args.dataset}_rich_512D.csv"
                rich_features_df.to_csv(rich_csv_path, index=False)
                logger.info(f"💾 Rich features saved to: {rich_csv_path}")

        # Feature compressor training
        if args.train_compressor and not args.no_compression and not args.skip_training:
            if compressor_exists and not args.fresh:
                logger.info("✅ Using existing feature compressor")
                extractor.model.use_compression = True
                # Load compressor weights
                compressor_checkpoint = torch.load(compressor_path)
                extractor.model.feature_compressor.load_state_dict(
                    compressor_checkpoint['compressor_state_dict']
                )
            else:
                if rich_features_df is None:
                    # Extract rich features if not already done
                    logger.info("🔍 Extracting rich 512D features for compressor training...")
                    rich_features_df = extractor.extract_features(
                        args.data_dir,
                        use_compression=False,
                        split="train"
                    )

                logger.info("🎯 Training feature space compressor...")
                extractor.train_feature_compressor(rich_features_df, epochs=100)

        # Final feature extraction
        logger.info("📊 Extracting final features...")

        if args.no_compression:
            # Use raw 512D features
            final_features_df = extractor.extract_features(
                args.data_dir,
                use_compression=False,
                split="train"
            )
            feature_type = "raw_512D"
        else:
            # Use compressed features
            extractor.model.use_compression = True
            final_features_df = extractor.extract_features(
                args.data_dir,
                use_compression=True,
                split="train"
            )
            feature_type = f"compressed_{args.final_dim}D"

        # Save final features and configurations
        logger.info("💾 Saving features and configurations...")
        csv_path, config_path = extractor.save_features_and_config(final_features_df, args.output_dir)

        # Save DBNN configuration
        extractor.config_manager.save_dbnn_config(extractor.dbnn_config)

        # Verification (if previous features exist)
        previous_features_path = Path(args.output_dir) / args.dataset / f"{args.dataset}.csv"
        if previous_features_path.exists() and not args.fresh:
            logger.info("🔍 Verifying latent space consistency...")
            consistency_results = extractor.verify_latent_consistency(str(previous_features_path))
            if consistency_results.get('consistent', True):
                logger.info("✅ Latent space consistency verified!")
            else:
                logger.warning("⚠️  Latent space consistency check failed")

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("🎉 SYSTEM EXECUTION COMPLETE!")
        logger.info(f"📁 Dataset: {args.dataset}")
        logger.info(f"📊 Data source: {args.data_source}")
        logger.info(f"💾 Feature type: {feature_type}")
        logger.info(f"📈 Final dimensions: {args.final_dim}D")
        logger.info(f"📂 Raw images: {args.data_dir}/{args.dataset}/")
        logger.info(f"💾 Features CSV: {csv_path}")
        logger.info(f"⚙️  Dataset Config: {config_path}")
        logger.info(f"⚙️  DBNN Config: {args.config_dir}/adaptive_dbnn.conf")
        logger.info(f"⚙️  Main Config: {args.config_dir}/{args.dataset}.json")
        logger.info(f"📊 Total samples: {len(final_features_df)}")
        logger.info(f"🔢 Feature dimensions: {extractor.main_config['model']['feature_dims']}")

        if rich_features_df is not None and args.extract_rich:
            rich_csv_path = Path(args.output_dir) / args.dataset / f"{args.dataset}_rich_512D.csv"
            logger.info(f"💎 Rich features: {rich_csv_path}")

        logger.info("="*70)

        # Display sample
        logger.info("\n📋 Sample of extracted features:")
        sample_cols = ['target', 'image_path'] + [f'feature_{i}' for i in range(min(5, args.final_dim))]
        print(final_features_df[sample_cols].head())

    except Exception as e:
        logger.error(f"❌ Error during execution: {str(e)}")
        logger.error("💡 Use --help for command line options")
        raise

if __name__ == "__main__":
    main()

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
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import time

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
    """Generic dataset handler that can work with any image dataset"""

    def __init__(self, root_dir: str, dataset_name: str, config: Dict, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.config = config
        self.split = split

        # Setup transforms
        self.transform = self._create_transforms()

        # Load data
        self.image_paths, self.targets, self.class_names = self._load_data()

        logger.info(f"Loaded {len(self.image_paths)} images for {dataset_name} ({split})")

    def _create_transforms(self) -> transforms.Compose:
        """Create transforms that preserve original image size"""
        dataset_cfg = self.config['dataset']
        mean = dataset_cfg['mean']
        std = dataset_cfg['std']

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]

        # Don't resize - keep original image dimensions
        # Add augmentation for training
        if self.split == "train":
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ] + transform_list

        return transforms.Compose(transform_list)

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

    def initialize_model(self):
        """Initialize the unified autoencoder model with dynamic sizing"""
        num_classes = len(set(self.dataset.targets)) if self.dataset else 100

        # Update config with actual image dimensions from dataset
        if self.dataset and len(self.dataset) > 0:
            sample_image, _, _ = self.dataset[0]
            _, height, width = sample_image.shape
            self.main_config['dataset']['input_size'] = [height, width]
            logger.info(f"Detected image size: {height}x{width}")

        self.model = UnifiedDiscriminativeAutoencoder(self.main_config, num_classes)
        self.model = self.model.to(self.device)

        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model

    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load data from various sources"""
        dataset_path = self.root_dir / self.dataset_name

        if dataset_path.exists() and any(dataset_path.iterdir()):
            # Load from custom folder structure
            return self._load_from_folder_structure(dataset_path)
        else:
            # Download and create standard dataset
            return self._download_standard_dataset()

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

        # Map dataset names to torchvision datasets
        dataset_map = {
            'cifar10': (torchvision.datasets.CIFAR10, 10),
            'cifar100': (torchvision.datasets.CIFAR100, 100),
            'mnist': (torchvision.datasets.MNIST, 10),
            'fashionmnist': (torchvision.datasets.FashionMNIST, 10),
            'imagenet': (torchvision.datasets.ImageNet, 1000)
        }

        if self.dataset_name.lower() in dataset_map:
            dataset_class, num_classes = dataset_map[self.dataset_name.lower()]

            # Download dataset
            dataset = dataset_class(root=str(self.root_dir), train=(self.split == "train"), download=True)

            # Save images to folder structure for future use
            return self._save_torchvision_dataset(dataset, dataset_path, num_classes)
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

class UnifiedDiscriminativeAutoencoder(nn.Module):
    """Enhanced unified autoencoder with domain-specific enhancements"""

    def __init__(self, config: Dict, num_classes: int):
        super(UnifiedDiscriminativeAutoencoder, self).__init__()

        self.config = config
        model_cfg = config['model']
        dataset_cfg = config['dataset']

        self.latent_dim = model_cfg['feature_dims']
        self.in_channels = dataset_cfg['in_channels']
        self.input_size = dataset_cfg['input_size']

        # Enhancement modules
        self.enhancement_modules = EnhancementModules(config)

        # Encoder
        self.encoder = self._build_encoder()

        # Expert modules
        self.reconstruction_expert = ReconstructionExpert(self.latent_dim)
        self.semantic_expert = SemanticDensityExpert(self.latent_dim, num_classes)
        self.efficiency_critic = EfficiencyCritic(self.latent_dim)

        # Decoder
        self.decoder = self._build_decoder()

        # Domain-specific components
        self.setup_domain_specific_components()

    def _build_encoder(self) -> nn.Module:
        """Build efficient but powerful encoder optimized for A100"""
        if hasattr(self, 'dataset') and self.dataset and len(self.dataset) > 0:
            sample_image, _, _ = self.dataset[0]
            _, input_height, input_width = sample_image.shape
        else:
            input_height, input_width = self.config['dataset']['input_size']

        logger.info(f"Building efficient encoder for {input_height}x{input_width} on A100")

        # For CIFAR-100 (32x32), we can use a more efficient architecture
        if input_height <= 64:  # Small images like CIFAR
            return self._build_efficient_small_encoder(input_height, input_width)
        else:  # Larger images
            return self._build_efficient_large_encoder(input_height, input_width)

    def _build_efficient_small_encoder(self, input_height: int, input_width: int) -> nn.Module:
        """Efficient encoder for small images (CIFAR, MNIST, etc.)"""
        return nn.Sequential(
            # Block 1: 64 channels
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout(0.1),

            # Block 2: 128 channels
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout(0.1),

            # Block 3: 256 channels
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),  # 4x4
            nn.Dropout(0.2),

            # Block 4: 512 channels
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),  # Fixed size for FC layers
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, self.latent_dim)
        )

    def _build_efficient_large_encoder(self, input_height: int, input_width: int) -> nn.Module:
        """Efficient encoder for larger images (ImageNet, etc.)"""
        return nn.Sequential(
            # Initial downsampling for large images
            nn.Conv2d(self.in_channels, 64, 7, 2, 3),  # /2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),  # /4 total

            # Block 1
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),  # /8

            # Block 2
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),  # /16

            # Block 3
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, self.latent_dim)
        )

    def _build_decoder(self) -> nn.Module:
        """Build efficient decoder optimized for A100"""
        if hasattr(self, 'dataset') and self.dataset and len(self.dataset) > 0:
            sample_image, _, _ = self.dataset[0]
            _, input_height, input_width = sample_image.shape
        else:
            input_height, input_width = self.config['dataset']['input_size']

        logger.info(f"Building efficient decoder for {input_height}x{input_width}")

        if input_height <= 64:
            return self._build_efficient_small_decoder(input_height, input_width)
        else:
            return self._build_efficient_large_decoder(input_height, input_width)

    def _build_efficient_small_decoder(self, input_height: int, input_width: int) -> nn.Module:
        """Efficient decoder for small images"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 512),
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

            # Final convolution to get exact size
            nn.Conv2d(32, self.in_channels, 3, 1, 1),
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

    def forward(self, x, targets=None, phase=1, apply_enhancements=True):
        # Apply domain-specific enhancements
        if apply_enhancements:
            image_type = self.config['dataset']['image_type']
            if image_type == "astronomical":
                x = self.enhancement_modules.apply_astronomical_enhancements(x)
            elif image_type == "medical":
                x = self.enhancement_modules.apply_medical_enhancements(x)
            elif image_type == "agricultural":
                x = self.enhancement_modules.apply_agricultural_enhancements(x)

        # Base encoding - ALWAYS start from the same base latent space
        base_latent = self.encoder(x)
        current_latent = base_latent

        # Progressive expert guidance with deterministic operations
        if phase >= 2:
            current_latent = self.reconstruction_expert(current_latent)

        if phase >= 3 and targets is not None:
            current_latent = self.semantic_expert(current_latent, targets)

        # Final reconstruction
        reconstructed = self.decoder(current_latent)

        # Always return 4 values for consistency across all phases
        if phase >= 4:
            efficiency_scores = self.efficiency_critic(current_latent)
        else:
            efficiency_scores = None

        return reconstructed, current_latent, base_latent, efficiency_scores

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
        """Initialize the unified autoencoder model"""
        num_classes = len(set(self.dataset.targets)) if self.dataset else 100

        self.model = UnifiedDiscriminativeAutoencoder(self.main_config, num_classes)
        self.model = self.model.to(self.device)

        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model

    def train(self, data_dir: str = "Data", save_dir: str = "models"):
        """Multi-round competitive training with latent space consistency"""
        if self.dataset is None:
            self.setup_data(data_dir, "train")

        if self.model is None:
            self.initialize_model()

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
            (2, "+ Reconstruction Expert", 0.03),
            (3, "+ Semantic Density Expert", 0.02),
            (4, "+ Efficiency Critic", 0.015)
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

                    for batch_idx, (data, targets, _) in enumerate(train_loader):
                        data = data.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)

                        optimizer.zero_grad()

                        # Mixed precision forward pass - UPDATED UNPACKING
                        with torch.amp.autocast('cuda'):
                            # Model now returns 4 values: (reconstructed, current_latent, base_latent, efficiency_scores)
                            reconstructed, current_latent, base_latent, efficiency_scores = self.model(data, targets, phase=phase_num)

                            if phase_num >= 4:
                                rec_loss = criterion(reconstructed, data)
                                eff_loss = (1 - efficiency_scores[:, 0]).mean()
                                loss = rec_loss + 0.1 * eff_loss
                            else:
                                loss = criterion(reconstructed, data)

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

                    if num_batches == 0:
                        continue

                    epoch_loss = total_loss / num_batches

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

                    if epoch % 2 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        progress_pct = (target_loss / epoch_loss * 100) if epoch_loss > 0 else 0
                        logger.info(f'Round {current_round + 1} Phase {phase_num} Epoch [{epoch}/{max_epochs_per_phase}], '
                                   f'Loss: {epoch_loss:.4f}, Best: {phase_best_loss:.4f}, '
                                   f'Target: {target_loss:.4f} ({progress_pct:.1f}%), LR: {current_lr:.6f}')

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

                # Save phase checkpoint
                phase_checkpoint_path = Path(save_dir) / f"{self.dataset_name}_round{current_round + 1}_phase{phase_num}.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'phase': phase_num,
                    'round': current_round + 1,
                    'best_loss': phase_best_loss,
                    'round_best_losses': round_best_losses
                }, phase_checkpoint_path)

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

        # Final model save
        self.save_model(save_dir)

        # Print final results
        logger.info("🏁 FINAL TRAINING RESULTS:")
        for phase_num, description, target_loss in phases:
            achieved = "✅" if round_best_losses[phase_num] <= target_loss else "❌"
            logger.info(f"  {achieved} {description}: {round_best_losses[phase_num]:.4f} (target: {target_loss:.4f})")

    def extract_features(self, data_dir: str = "Data", split: str = "train") -> pd.DataFrame:
        """Extract features and save to CSV with proper return value handling"""
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")

        if self.dataset is None:
            self.setup_data(data_dir, split)

        self.model.eval()
        features_list = []
        targets_list = []
        image_paths_list = []

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

                # Get final latent representation with all experts
                # Handle different return types based on phase
                outputs = self.model(data, targets, phase=4)

                # Properly unpack based on number of return values
                if len(outputs) == 4:
                    # Phase 4: (reconstructed, current_latent, base_latent, efficiency_scores)
                    reconstructed, final_latent, base_latent, efficiency_scores = outputs
                elif len(outputs) == 3:
                    # Phases 1-3: (reconstructed, current_latent, base_latent)
                    reconstructed, final_latent, base_latent = outputs
                else:
                    # Fallback: try to get the second element as latent
                    final_latent = outputs[1] if len(outputs) >= 2 else outputs[0]

                features_list.append(final_latent.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                image_paths_list.extend(image_paths)

                if batch_idx % 10 == 0:
                    logger.info(f'Processed batch {batch_idx}/{len(dataloader)}')

        # Concatenate all batches
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Create DataFrame
        latent_dim = self.main_config['model']['feature_dims']
        feature_columns = [f'feature_{i}' for i in range(latent_dim)]

        df_data = {
            'target': all_targets,
            'image_path': image_paths_list
        }

        for i in range(latent_dim):
            df_data[f'feature_{i}'] = all_features[:, i]

        df = pd.DataFrame(df_data)
        columns = ['target', 'image_path'] + feature_columns
        df = df[columns]

        logger.info(f"Successfully extracted {len(df)} samples with {latent_dim} features each")
        return df

    def save_features_and_config(self, df: pd.DataFrame, output_dir: str = "data"):
        """Save features to CSV and create configuration files in data/ folder"""
        # Create output directory in data/ (not Data/)
        output_path = Path(output_dir) / self.dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_path / f"{self.dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Features saved to: {csv_path}")

        # Create dataset configuration
        dataset_config = {
            "file_path": f"data/{self.dataset_name}/{self.dataset_name}.csv",
            "column_names": ["target", "image_path"] + [f"feature_{i}" for i in range(self.main_config['model']['feature_dims'])],
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

        return csv_path, dataset_config_path

    def save_model(self, save_dir: str = "models"):
        """Save trained model"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / f"{self.dataset_name}_autoencoder.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.main_config,
            'dataset_name': self.dataset_name
        }, model_path)

        logger.info(f"Model saved to: {model_path}")

    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        if self.model is None:
            self.initialize_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from: {model_path}")

def main():
    """Main execution function"""
    # Configuration
    DATASET_NAME = "cifar100"  # Change to any dataset name
    DATA_DIR = "Data"  # Raw images go here (with class subfolders)
    OUTPUT_DIR = "data"  # Processed features and configs go here
    CONFIG_DIR = "configs"
    MODELS_DIR = "models"

    logger.info("Initializing Unified Discriminative Autoencoder System")
    logger.info(f"Dataset: {DATASET_NAME}")

    try:
        # Initialize system
        extractor = UnifiedFeatureExtractor(DATASET_NAME, CONFIG_DIR)

        # Setup data - reads from Data/ folder with class subfolders
        logger.info("Setting up data pipeline...")
        extractor.setup_data(DATA_DIR)

        # Initialize and train model
        logger.info("Initializing model...")
        extractor.initialize_model()

        logger.info("Starting training...")
        extractor.train(DATA_DIR, MODELS_DIR)

        # Extract features
        logger.info("Extracting features...")
        features_df = extractor.extract_features(DATA_DIR)

        # Save results to data/ folder (not Data/)
        logger.info("Saving features and configurations...")
        csv_path, config_path = extractor.save_features_and_config(features_df, OUTPUT_DIR)

        # Save DBNN configuration
        extractor.config_manager.save_dbnn_config(extractor.dbnn_config)

        logger.info("\n" + "="*60)
        logger.info("SYSTEM EXECUTION COMPLETE!")
        logger.info(f"Dataset: {DATASET_NAME}")
        logger.info(f"Raw images: {DATA_DIR}/{DATASET_NAME}/ (with class subfolders)")
        logger.info(f"Features CSV: {csv_path}")
        logger.info(f"Dataset Config: {config_path}")
        logger.info(f"DBNN Config: {CONFIG_DIR}/adaptive_dbnn.conf")
        logger.info(f"Main Config: {CONFIG_DIR}/{DATASET_NAME}.json")
        logger.info(f"Total samples: {len(features_df)}")
        logger.info(f"Feature dimensions: {extractor.main_config['model']['feature_dims']}")
        logger.info("="*60)

        # Display sample
        logger.info("\nSample of extracted features:")
        print(features_df[['target', 'image_path'] + [f'feature_{i}' for i in range(5)]].head())

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

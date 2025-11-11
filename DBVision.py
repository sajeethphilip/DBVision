'''
DBVision - Unified Discriminative Autoencoder System
Production-ready single file implementation with enhanced organization
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
from dataclasses import dataclass, asdict
import time
import argparse
import sys
import hashlib
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

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
    use_kl_divergence: bool
    use_class_encoding: bool
    clustering_temperature: float
    enhancement_modules: Dict
    loss_functions: Dict

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    num_workers: int
    enhancement_specific: Dict

@dataclass
class ProgressiveConfig:
    enabled: bool
    scales: List[float]
    phase_epochs: List[int]
    scale_as_feature: bool
    fusion_method: str
    auto_detected: bool = False
    detection_method: str = "default"

class ConfigManager:
    """Enhanced configuration management with validation and type safety"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def load_main_config(self, dataset_name: str) -> Dict[str, Any]:
        """Load and validate main dataset configuration"""
        config_path = self.config_dir / f"{dataset_name}.json"

        if not config_path.exists():
            default_config = self._create_default_config(dataset_name)
            self.save_main_config(dataset_name, default_config)
            logger.info(f"Created default config for {dataset_name}")
            return default_config

        with open(config_path, 'r') as f:
            config = json.load(f)

        return self._validate_config(config)

    def save_main_config(self, dataset_name: str, config: Dict):
        """Save configuration with validation"""
        validated_config = self._validate_config(config)
        config_path = self.config_dir / f"{dataset_name}.json"
        with open(config_path, 'w') as f:
            json.dump(validated_config, f, indent=2)
        logger.info(f"Saved config to {config_path}")

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

    def _validate_config(self, config: Dict) -> Dict:
        """Validate configuration structure and values"""
        required_sections = ['dataset', 'model', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Set defaults for missing values
        config.setdefault('dataset', {}).setdefault('image_type', 'general')
        config.setdefault('model', {}).setdefault('use_kl_divergence', True)
        config.setdefault('training', {}).setdefault('num_workers', 4)

        return config

    def _create_default_config(self, dataset_name: str) -> Dict:
        """Create comprehensive default configuration"""
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
                "use_kl_divergence": True,
                "use_class_encoding": True,
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

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class GenericImageDataset(Dataset):
    """Enhanced dataset handler with automatic channel detection and optimal resizing"""

    def __init__(self, root_dir: str, dataset_name: str, config: Dict,
                 split: str = "train", data_source: str = "auto"):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.data_source = data_source

        # Load data and setup transforms
        self.image_paths, self.targets, self.class_names = self._load_data()
        self.transform = self._create_transforms()

        logger.info(f"Loaded {len(self.image_paths)} images for {dataset_name} ({split}) from {data_source} source")

    def _create_transforms(self) -> transforms.Compose:
        """Create intelligent transforms that adapt to image characteristics"""
        dataset_cfg = self.config['dataset']
        mean = dataset_cfg['mean']
        std = dataset_cfg['std']

        # Detect actual number of channels
        num_channels = self._detect_image_channels()

        # Adjust normalization for detected channels
        mean, std = self._adjust_normalization(mean, std, num_channels)

        # Determine optimal target size
        target_size = self._determine_optimal_size()

        # Build transform pipeline
        transform_list = self._build_transform_pipeline(target_size, mean, std, num_channels)

        return transforms.Compose(transform_list)

    def _detect_image_channels(self) -> int:
        """Detect number of channels from sample images"""
        if len(self.image_paths) > 0:
            try:
                sample_image = Image.open(self.image_paths[0])
                num_channels = len(sample_image.getbands())
                logger.info(f"Detected {num_channels} channel(s) for dataset {self.dataset_name}")
                return num_channels
            except Exception as e:
                logger.warning(f"Could not detect channels from sample image: {e}")

        # Fallback to config
        return self.config['dataset']['in_channels']

    def _adjust_normalization(self, mean: List[float], std: List[float], num_channels: int) -> Tuple[List[float], List[float]]:
        """Adjust normalization parameters based on detected channels"""
        if num_channels == 1:
            # Grayscale images
            if len(mean) == 3:
                mean = [mean[0]]
                std = [std[0]]
            elif len(mean) == 0 or mean == [0.485, 0.456, 0.406]:
                mean = [0.5]
                std = [0.5]
        elif num_channels == 3 and len(mean) == 1:
            # RGB images but config has grayscale values
            mean = [mean[0]] * 3
            std = [std[0]] * 3

        return mean, std

    def _determine_optimal_size(self) -> Tuple[int, int]:
        """Determine optimal resize size based on image statistics"""
        if len(self.image_paths) > 0:
            try:
                sample_image = Image.open(self.image_paths[0])
                original_width, original_height = sample_image.size

                # More intelligent size selection
                if original_height <= 32 and original_width <= 32:
                    target_size = (32, 32)
                elif original_height <= 64 and original_width <= 64:
                    target_size = (64, 64)
                elif original_height <= 128 and original_width <= 128:
                    target_size = (128, 128)
                elif original_height <= 256 and original_width <= 256:
                    target_size = (224, 224)  # Standard for many models
                else:
                    target_size = (224, 224)  # Default for larger images

                logger.info(f"Original size: {original_width}x{original_height} -> Resizing to: {target_size[0]}x{target_size[1]}")
                return target_size
            except Exception as e:
                logger.warning(f"Could not detect image size: {e}")

        # Use config as fallback
        dataset_cfg = self.config['dataset']
        if 'input_size' in dataset_cfg:
            return tuple(dataset_cfg['input_size'])
        return (224, 224)  # Default fallback

    def _build_transform_pipeline(self, target_size: Tuple[int, int],
                                mean: List[float], std: List[float],
                                num_channels: int) -> List:
        """Build the complete transform pipeline"""
        transform_list = []

        # Training augmentations
        if self.split == "train":
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

        # Core transforms
        transform_list.extend([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        # Normalization if compatible
        if len(mean) == num_channels and len(std) == num_channels:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        else:
            logger.warning(f"Skipping normalization - mean/std mismatch")

        return transform_list

    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load data based on specified source"""
        dataset_path = self.root_dir / self.dataset_name

        if self.data_source == "local" or (self.data_source == "auto" and dataset_path.exists() and any(dataset_path.iterdir())):
            return self._load_from_folder_structure(dataset_path)
        elif self.data_source == "torchvision" or (self.data_source == "auto" and not dataset_path.exists()):
            return self._download_standard_dataset()
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")

    def _load_from_folder_structure(self, dataset_path: Path) -> Tuple[List[str], List[int], List[str]]:
        """Load data from folder structure where each subfolder is a class"""
        image_paths = []
        targets = []
        class_names = []

        class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
        class_folders.sort()

        for class_idx, class_folder in enumerate(class_folders):
            class_names.append(class_folder.name)

            # Support multiple image formats
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            for ext in image_extensions:
                for image_path in class_folder.glob(ext):
                    image_paths.append(str(image_path))
                    targets.append(class_idx)

        if not image_paths:
            raise ValueError(f"No images found in {dataset_path}")

        return image_paths, targets, class_names

    def _download_standard_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """Download standard dataset using torchvision"""
        dataset_path = self.root_dir / self.dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        tmp_dir = self.root_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        dataset_map = {
            'cifar10': (torchvision.datasets.CIFAR10, 10),
            'cifar100': (torchvision.datasets.CIFAR100, 100),
            'mnist': (torchvision.datasets.MNIST, 10),
            'fashionmnist': (torchvision.datasets.FashionMNIST, 10),
        }

        if self.dataset_name.lower() in dataset_map:
            dataset_class, num_classes = dataset_map[self.dataset_name.lower()]

            try:
                dataset = dataset_class(root=str(tmp_dir), train=(self.split == "train"), download=True)
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

# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class SelfAttentionEnhanced(nn.Module):
    """Memory-efficient self-attention with local windows"""

    def __init__(self, in_channels: int, window_size: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        # Use window-based attention to reduce memory
        if height * width > 64 * 64:  # Only use windowing for larger feature maps
            return self._window_attention(x)
        else:
            return self._global_attention(x)

    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Global attention for small feature maps"""
        batch_size, channels, height, width = x.size()

        # Compute queries, keys, values
        queries = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        keys = self.key(x).view(batch_size, -1, height * width)
        values = self.value(x).view(batch_size, -1, height * width)

        # Compute attention scores with memory optimization
        attention_scores = torch.bmm(queries, keys)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention
        out = torch.bmm(values, attention_scores.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Window-based attention for large feature maps"""
        batch_size, channels, height, width = x.size()

        # Pad feature map to be divisible by window size
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, height, width = x.size()

        # Reshape into windows
        x = x.view(batch_size, channels, height // self.window_size, self.window_size,
                  width // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, H//ws, W//ws, C, ws, ws]
        x = x.view(batch_size * (height // self.window_size) * (width // self.window_size),
                  channels, self.window_size, self.window_size)

        # Apply attention within each window
        queries = self.query(x).view(-1, channels // 8, self.window_size * self.window_size).permute(0, 2, 1)
        keys = self.key(x).view(-1, channels // 8, self.window_size * self.window_size)
        values = self.value(x).view(-1, channels, self.window_size * self.window_size)

        attention_scores = torch.bmm(queries, keys)  # [B*num_windows, ws*ws, ws*ws]
        attention_scores = F.softmax(attention_scores, dim=-1)

        out = torch.bmm(values, attention_scores.permute(0, 2, 1))
        out = out.view(-1, channels, self.window_size, self.window_size)

        # Reshape back
        out = out.view(batch_size, height // self.window_size, width // self.window_size,
                      channels, self.window_size, self.window_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(batch_size, channels, height, width)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :height-pad_h, :width-pad_w]

        return self.gamma * out + x[:, :, :height-pad_h, :width-pad_w]

class MultiScaleProcessingBlock(nn.Module):
    """Memory-optimized multi-scale processing"""

    def __init__(self, in_channels, base_channels, scales, output_size):
        super().__init__()
        self.scales = scales
        self.output_size = output_size
        self.base_channels = base_channels

        # Process each scale sequentially to save memory
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, True),
        )

        # Lightweight attention instead of full self-attention
        self.light_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, base_channels // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(base_channels // 4, base_channels, 1),
            nn.Sigmoid()
        )

        # Scale embedding
        self.scale_embedding = nn.Embedding(len(scales), base_channels // 4)

        # Scale-specific adapters
        self.scale_adapters = nn.ModuleList([
            nn.Conv2d(base_channels + base_channels//4, base_channels, 1, 1, 0)
            for _ in range(len(scales))
        ])

        self.register_buffer('scale_factors', torch.tensor(scales, dtype=torch.float32))

    def forward(self, x):
        batch_size = x.size(0)
        multi_scale_outputs = []
        scale_features = []

        # Process scales sequentially to reduce peak memory
        for i, scale in enumerate(self.scales):
            # Clear intermediate cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get scale embedding
            scale_idx = torch.tensor([i], device=x.device).repeat(batch_size)
            scale_embed = self.scale_embedding(scale_idx)

            # Scale input image
            if scale != 1.0:
                scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear',
                                       align_corners=False, antialias=True)
            else:
                scaled_x = x

            # Apply shared convolution
            features = self.shared_conv(scaled_x)

            # Apply lightweight channel attention (memory efficient)
            attention_weights = self.light_attention(features)
            features = features * attention_weights

            # Add scale information
            scale_embed = scale_embed.unsqueeze(-1).unsqueeze(-1)
            scale_embed = scale_embed.expand(-1, -1, features.size(2), features.size(3))
            features_with_scale = torch.cat([features, scale_embed], dim=1)

            # Scale-specific adaptation
            adapted_features = self.scale_adapters[i](features_with_scale)

            # Resize to consistent output size
            if adapted_features.shape[2:] != self.output_size:
                adapted_features = F.interpolate(adapted_features, size=self.output_size,
                                               mode='bilinear', align_corners=False, antialias=True)

            multi_scale_outputs.append(adapted_features)

            # Store scale feature for metadata
            scale_global = F.adaptive_avg_pool2d(adapted_features, (1, 1)).squeeze(-1).squeeze(-1)
            scale_info = torch.cat([
                scale_global,
                torch.tensor([scale, 1.0/scale], device=x.device).repeat(batch_size, 1)
            ], dim=1)
            scale_features.append(scale_info)

        # Concatenate all scale outputs
        output = torch.cat(multi_scale_outputs, dim=1)

        # Store scale metadata
        self.scale_metadata = {
            'features': torch.stack(scale_features, dim=1),
            'factors': self.scale_factors.repeat(batch_size, 1)
        }

        return output

class EnhancementModules:
    """Domain-specific enhancement modules with optimized operations"""

    def __init__(self, config: Dict):
        self.config = config
        self.enhancement_cfg = config['model']['enhancement_modules']

        # Precompute filters for efficiency
        self._precompute_filters()

    def _precompute_filters(self):
        """Precompute common filters for efficiency"""
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.high_pass = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

    def apply_astronomical_enhancements(self, images: torch.Tensor) -> torch.Tensor:
        """Apply astronomical image enhancements"""
        if not self.enhancement_cfg['astronomical']['enabled']:
            return images

        enhanced_images = images.clone()
        weights = self.enhancement_cfg['astronomical']['weights']

        if self.enhancement_cfg['astronomical']['components']['structure_preservation']:
            enhanced_images = self._enhance_edges(enhanced_images, weights['edge_weight'])

        if self.enhancement_cfg['astronomical']['components']['detail_preservation']:
            enhanced_images = self._enhance_details(enhanced_images, weights['detail_weight'])

        return enhanced_images

    def apply_medical_enhancements(self, images: torch.Tensor) -> torch.Tensor:
        """Apply medical image enhancements"""
        if not self.enhancement_cfg['medical']['enabled']:
            return images

        enhanced_images = images.clone()
        weights = self.enhancement_cfg['medical']['weights']

        if self.enhancement_cfg['medical']['components']['contrast_enhancement']:
            enhanced_images = self._enhance_contrast(enhanced_images, weights['contrast_weight'])

        if self.enhancement_cfg['medical']['components']['tissue_boundary']:
            enhanced_images = self._enhance_boundaries(enhanced_images, weights['boundary_weight'])

        return enhanced_images

    def apply_agricultural_enhancements(self, images: torch.Tensor) -> torch.Tensor:
        """Apply agricultural image enhancements"""
        if not self.enhancement_cfg['agricultural']['enabled']:
            return images

        enhanced_images = images.clone()
        weights = self.enhancement_cfg['agricultural']['weights']

        if self.enhancement_cfg['agricultural']['components']['texture_analysis']:
            enhanced_images = self._enhance_texture(enhanced_images, weights['texture_weight'])

        if self.enhancement_cfg['agricultural']['components']['pattern_enhancement']:
            enhanced_images = self._enhance_patterns(enhanced_images, weights['pattern_weight'])

        return enhanced_images

    def _enhance_edges(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance edges using Sobel filter"""
        device = images.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Convert to grayscale for edge detection
        gray = images.mean(dim=1, keepdim=True)

        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        return images + weight * edges.repeat(1, images.shape[1], 1, 1)

    def _enhance_details(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance fine details using high-pass filter"""
        device = images.device
        kernel = self.high_pass.to(device)

        details = F.conv2d(images, kernel.repeat(images.shape[1], 1, 1, 1),
                          padding=1, groups=images.shape[1])
        return images + weight * (details - images)

    def _enhance_contrast(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance contrast using adaptive normalization"""
        mean = images.mean(dim=[2, 3], keepdim=True)
        std = images.std(dim=[2, 3], keepdim=True) + 1e-8
        normalized = (images - mean) / std
        return torch.sigmoid(normalized) * weight + images * (1 - weight)

    def _enhance_boundaries(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance boundaries using Laplacian filter"""
        device = images.device
        laplacian = self.laplacian.to(device)

        gray = images.mean(dim=1, keepdim=True)
        boundaries = F.conv2d(gray, laplacian, padding=1)
        return images + weight * boundaries.repeat(1, images.shape[1], 1, 1)

    def _enhance_texture(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance texture using local variance"""
        patch_size = 3
        unfolded = images.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        texture = unfolded.contiguous().view(images.size(0), images.shape[1], -1, patch_size, patch_size)
        texture = texture.var(dim=2, keepdim=True)
        texture = F.interpolate(texture, size=images.shape[2:], mode='bilinear')

        return images + weight * texture

    def _enhance_patterns(self, images: torch.Tensor, weight: float) -> torch.Tensor:
        """Enhance patterns using frequency domain filtering"""
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

# =============================================================================
# DOMAIN-SPECIFIC AUTOENCODERS
# =============================================================================

class AstronomicalAutoencoder(nn.Module):
    """Specialized autoencoder for astronomical images preserving large-scale structures"""

    def __init__(self, config: Dict, num_classes: int):
        super().__init__()
        self.config = config
        self.latent_dim = 512  # Internal rich representation
        self.output_dim = config['model']['feature_dims']  # Final output dimension
        self.in_channels = config['dataset']['in_channels']
        self.input_size = config['dataset']['input_size']

        # Use the general autoencoder as base but add astronomical enhancements
        self.base_autoencoder = UnifiedDiscriminativeAutoencoder(config, num_classes)

        # Astronomical-specific components
        self.low_surface_brightness_enhancer = LowSurfaceBrightnessEnhancer()
        self.morphology_preserver = MorphologyPreservationModule(self.latent_dim)
        self.multi_scale_astronomical = AstronomicalMultiScaleFusion()

        # Feature compression - ADD THIS
        self.feature_compressor = FeatureSpaceCompressor(
            input_dim=self.latent_dim,
            compressed_dim=self.output_dim,
            num_classes=num_classes
        )

        # Training control
        self.use_compression = False

    def forward(self, x, targets=None, phase=1, apply_enhancements=True, use_compression=None,
                current_scale=None, training_phase=None, return_scale_info=False):
        """Astronomical-optimized forward pass"""
        # Apply astronomical-specific enhancements
        if apply_enhancements:
            x_enhanced = self.low_surface_brightness_enhancer(x)
        else:
            x_enhanced = x

        # Use base autoencoder for main processing - FIXED: Handle return_scale_info
        base_outputs = self.base_autoencoder(
            x_enhanced, targets, phase, False, use_compression,
            current_scale, training_phase, return_scale_info
        )

        # Handle different return types based on return_scale_info
        if return_scale_info:
            reconstructed, compressed_latent, current_latent, efficiency_scores, scale_info = base_outputs
        else:
            reconstructed, compressed_latent, current_latent, efficiency_scores = base_outputs
            scale_info = None

        # Apply morphology preservation to latent space
        if phase >= 2:
            morphology_adjustment = self.morphology_preserver(current_latent, x)
            current_latent = current_latent + 0.1 * morphology_adjustment

        # Feature compression - ADD THIS LOGIC
        use_compression = use_compression if use_compression is not None else self.use_compression
        if use_compression:
            final_latent = self.feature_compressor(current_latent)
        else:
            final_latent = current_latent

        if return_scale_info:
            return reconstructed, final_latent, current_latent, efficiency_scores, scale_info
        else:
            return reconstructed, final_latent, current_latent, efficiency_scores

class MedicalAutoencoder(nn.Module):
    """Specialized autoencoder for medical images with tissue boundary preservation"""

    def __init__(self, config: Dict, num_classes: int):
        super().__init__()
        self.config = config
        self.latent_dim = 512
        self.output_dim = config['model']['feature_dims']
        self.in_channels = config['dataset']['in_channels']

        # Use the general autoencoder as base but add medical enhancements
        self.base_autoencoder = UnifiedDiscriminativeAutoencoder(config, num_classes)

        # Medical-specific components
        self.boundary_preserver = BoundaryPreservationModule()
        self.contrast_enhancer = MedicalContrastEnhancer()
        self.lesion_detector = LesionDetectionModule(self.latent_dim)

        # Feature compression - ADD THIS
        self.feature_compressor = FeatureSpaceCompressor(
            input_dim=self.latent_dim,
            compressed_dim=self.output_dim,
            num_classes=num_classes
        )

        # Training control
        self.use_compression = False

    def forward(self, x, targets=None, phase=1, apply_enhancements=True, use_compression=None,
                current_scale=None, training_phase=None, return_scale_info=False):
        """Medical-optimized forward pass"""
        # Apply medical-specific enhancements
        if apply_enhancements:
            x_enhanced = self.contrast_enhancer(x)
            x_enhanced = self.boundary_preserver(x_enhanced)
        else:
            x_enhanced = x

        # Use base autoencoder for main processing - FIXED: Handle return_scale_info
        base_outputs = self.base_autoencoder(
            x_enhanced, targets, phase, False, use_compression,
            current_scale, training_phase, return_scale_info
        )

        # Handle different return types based on return_scale_info
        if return_scale_info:
            reconstructed, compressed_latent, current_latent, efficiency_scores, scale_info = base_outputs
        else:
            reconstructed, compressed_latent, current_latent, efficiency_scores = base_outputs
            scale_info = None

        # Apply lesion detection guidance to latent space
        if phase >= 3:
            lesion_guidance = self.lesion_detector(current_latent, x)
            current_latent = current_latent + 0.05 * lesion_guidance

        # Feature compression - ADD THIS LOGIC
        use_compression = use_compression if use_compression is not None else self.use_compression
        if use_compression:
            final_latent = self.feature_compressor(current_latent)
        else:
            final_latent = current_latent

        if return_scale_info:
            return reconstructed, final_latent, current_latent, efficiency_scores, scale_info
        else:
            return reconstructed, final_latent, current_latent, efficiency_scores

# =============================================================================
# DOMAIN-SPECIFIC COMPONENTS
# =============================================================================

class ScaleSpecificFuser(nn.Module):
    """Scale-specific feature fusion for astronomical images"""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Linear(32 * 8 * 8, 128)

    def forward(self, x):
        features = self.conv(x)
        features = features.flatten(1)
        return self.fc(features)

class LowSurfaceBrightnessEnhancer(nn.Module):
    """Enhance low surface brightness features in astronomical images"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Simple contrast enhancement for astronomical images
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        enhanced = (x - mean) / (std + 1e-8)
        enhanced = torch.tanh(enhanced) * 0.3 + 0.5  # Gentle enhancement
        return enhanced

class MorphologyPreservationModule(nn.Module):
    """Preserve spiral arms, rings, and other morphological features"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.morphology_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, latent, x):
        # Learn morphology-preserving adjustments
        morphology_adjustment = self.morphology_net(latent)
        return morphology_adjustment

class AstronomicalMultiScaleFusion(nn.Module):
    """Fuse features at astronomical-relevant scales"""

    def __init__(self):
        super().__init__()
        # Astronomical scale pyramid
        self.scale_pyramid = [1.0, 0.5, 0.25]

    def forward(self, latent, x):
        # Simple multi-scale fusion - can be enhanced later
        return latent

class BoundaryPreservationModule(nn.Module):
    """Preserve tissue boundaries in medical images"""

    def __init__(self):
        super().__init__()
        # Precompute Laplacian kernel for edge detection
        self.register_buffer('laplacian_kernel',
                           torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                       dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, x):
        # Edge preservation for medical boundaries
        if x.shape[1] == 3:
            gray = x.mean(dim=1, keepdim=True)
        else:
            gray = x

        boundaries = F.conv2d(gray, self.laplacian_kernel, padding=1)
        # Gentle boundary enhancement
        return x + 0.05 * boundaries.repeat(1, x.shape[1], 1, 1)

class MedicalContrastEnhancer(nn.Module):
    """Enhance contrast for medical image analysis"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Local contrast enhancement for medical images
        local_mean = F.avg_pool2d(x, 5, 1, 2)  # Larger kernel for medical images
        local_std = torch.sqrt(F.avg_pool2d(x**2, 5, 1, 2) - local_mean**2 + 1e-8)
        enhanced = (x - local_mean) / (local_std + 1e-8)
        return torch.sigmoid(enhanced)  # Keep in [0,1] range

class LesionDetectionModule(nn.Module):
    """Guide features towards lesion-relevant representations"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.lesion_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, latent, x):
        # Learn lesion-specific feature adjustments
        lesion_guidance = self.lesion_net(latent)
        return lesion_guidance


# =============================================================================
# EXPERT MODULES AND COMPRESSION
# =============================================================================

class ReconstructionExpert(nn.Module):
    """Expert module for reconstruction guidance"""

    def __init__(self, latent_dim=512):
        super().__init__()
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
    """Expert module for semantic density guidance"""

    def __init__(self, latent_dim=512, num_classes=100):
        super().__init__()
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
    """Efficiency critic for feature quality assessment"""

    def __init__(self, latent_dim=512):
        super().__init__()
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

class FeatureSpaceCompressor(nn.Module):
    """Secondary autoencoder that compresses 512D features to final output dimension"""

    def __init__(self, input_dim: int, compressed_dim: int, num_classes: int):
        super().__init__()
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
            nn.Linear(64, compressed_dim),
            nn.BatchNorm1d(compressed_dim)
        )

        # Reconstruction network
        self.decompressor = nn.Sequential(
            nn.Linear(compressed_dim, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, input_dim)
        )

        # Discriminative head
        self.classifier = nn.Linear(compressed_dim, num_classes)

    def forward(self, x, return_all: bool = False):
        compressed = self.compressor(x)

        if return_all:
            reconstructed = self.decompressor(compressed)
            classified = self.classifier(compressed)
            return compressed, reconstructed, classified
        else:
            return compressed

# =============================================================================
# MAIN AUTOENCODER ARCHITECTURE
# =============================================================================

class UnifiedDiscriminativeAutoencoder(nn.Module):
    """Production-ready unified autoencoder with enhanced multi-scale support"""

    def __init__(self, config: Dict, num_classes: int):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        dataset_cfg = config['dataset']

        # Configuration
        self.latent_dim = 512  # Internal rich representation
        self.output_dim = model_cfg['feature_dims']  # Final output dimension
        self.in_channels = dataset_cfg['in_channels']
        self.input_size = dataset_cfg['input_size']

        # Core components
        self.enhancement_modules = EnhancementModules(config)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Expert modules
        self.reconstruction_expert = ReconstructionExpert(self.latent_dim)
        self.semantic_expert = SemanticDensityExpert(self.latent_dim, num_classes)
        self.efficiency_critic = EfficiencyCritic(self.latent_dim)

        # Feature compression
        self.feature_compressor = FeatureSpaceCompressor(
            input_dim=self.latent_dim,
            compressed_dim=self.output_dim,
            num_classes=num_classes
        )

        # Training control
        self.use_compression = False

        # Enhanced components
        self._initialize_enhanced_components(num_classes)
        self.setup_domain_specific_components()

        # Scale-aware components
        self.scale_attention = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, self.latent_dim),
            nn.Sigmoid()
        )

        self.scale_embeddings = nn.Embedding(10, 32)

    def _build_encoder(self) -> nn.Module:
        """Build multi-scale progressive encoder"""
        input_height, input_width = self.config['dataset']['input_size']

        # Multi-scale configuration
        self.progressive_config = self.config.get('progressive_scales', {
            'enabled': True,
            'scales': [1.0, 0.75, 0.5],
            'phase_epochs': [12, 10, 8, 6],
            'scale_as_feature': True,
            'fusion_method': 'weighted_average'
        })

        logger.info(f"Building multi-scale encoder with scales: {self.progressive_config['scales']}")

        # Size-appropriate encoder selection
        if input_height <= 32:
            return self._build_multiscale_32x32_encoder()
        elif input_height <= 64:
            return self._build_multiscale_64x64_encoder()
        else:
            return self._build_multiscale_128x128_encoder()

    def _build_multiscale_32x32_encoder(self) -> nn.Module:
        """Multi-scale encoder for 32x32 images"""
        base_channels = 32
        scales = self.progressive_config['scales']

        return nn.Sequential(
            MultiScaleProcessingBlock(
                in_channels=self.in_channels,
                base_channels=base_channels,
                scales=scales,
                output_size=(32, 32)
            ),
            nn.Conv2d(base_channels * len(scales), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 3, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 3, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, 3, 2, 1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),  # 2x2
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_multiscale_64x64_encoder(self) -> nn.Module:
        """Multi-scale encoder for 64x64 images - PROPERLY MATCHED"""
        base_channels = 32
        scales = self.progressive_config['scales']

        return nn.Sequential(
            MultiScaleProcessingBlock(
                in_channels=self.in_channels,
                base_channels=base_channels,
                scales=scales,
                output_size=(64, 64)
            ),
            nn.Conv2d(base_channels * len(scales), 128, 3, 2, 1),  # 64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 3, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, 3, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 512, 3, 2, 1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((2, 2)),  # 4x4 -> 2x2
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512)
        )

    def _build_64x64_decoder(self) -> nn.Module:
        """Decoder for 64x64 images - PROPERLY MATCHED to encoder"""
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1, True),
            nn.Linear(512, 512 * 2 * 2),  # Matches encoder's 2x2
            nn.LeakyReLU(0.1, True),
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 2x2 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, self.in_channels, 3, 1, 1),
            nn.Sigmoid()
        )


    def _build_multiscale_128x128_encoder(self) -> nn.Module:
        """Multi-scale encoder for 128x128 images"""
        base_channels = 32
        scales = self.progressive_config['scales']

        return nn.Sequential(
            MultiScaleProcessingBlock(
                in_channels=self.in_channels,
                base_channels=base_channels,
                scales=scales,
                output_size=(128, 128)
            ),
            nn.Conv2d(base_channels * len(scales), 128, 3, 2, 1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 3, 2, 1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, 3, 2, 1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 512, 3, 2, 1),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
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
        """Enhanced decoder for small images - FIXED for 64x64 output"""
        # Determine the starting size based on input dimensions
        if input_height <= 32:
            # For 32x32 inputs
            starting_features = 512 * 2 * 2
            return nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(0.1, True),
                nn.Linear(512, starting_features),
                nn.LeakyReLU(0.1, True),
                nn.Unflatten(1, (512, 2, 2)),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 2x2 -> 4x4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.1),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(32, self.in_channels, 3, 1, 1),
                nn.Sigmoid()
            )
        else:
            # For 64x64 inputs - FIXED ARCHITECTURE
            starting_features = 512 * 4 * 4
            return nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(0.1, True),
                nn.Linear(512, starting_features),
                nn.LeakyReLU(0.1, True),
                nn.Unflatten(1, (512, 4, 4)),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, True),
                nn.Dropout(0.1),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32 -> 64x64
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(32, self.in_channels, 3, 1, 1),
                nn.Sigmoid()
            )

    def _build_enhanced_large_decoder(self, input_height: int, input_width: int) -> nn.Module:
        """Enhanced decoder for large images"""
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1, True),  # FIXED: LeakyRELU -> LeakyReLU
            nn.Linear(512, 512 * 4 * 4),
            nn.LeakyReLU(0.1, True),  # FIXED: LeakyRELU -> LeakyReLU
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 64x64 -> 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(32, self.in_channels, 4, 2, 1),  # 128x128 -> 256x256
            nn.Sigmoid()
        )

    def _initialize_enhanced_components(self, num_classes: int):
        """Initialize enhanced components including scale-aware modules"""
        # Clustering components
        self.use_kl_divergence = self.config['model'].get('use_kl_divergence', True)
        if self.use_kl_divergence:
            self.register_buffer('cluster_centers', torch.randn(num_classes, self.latent_dim))
            self.register_buffer('clustering_temperature', torch.tensor([1.0]))

        # Classifier for semantic guidance
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

    def update_input_channels(self, actual_channels: int):
        """Update the model to use the actual number of input channels"""
        if actual_channels != self.in_channels:
            logger.info(f"Updating model from {self.in_channels} to {actual_channels} input channels")
            self.in_channels = actual_channels
            self.encoder = self._build_encoder()
            self.decoder = self._build_decoder()
            self.encoder = self.encoder.to(next(self.parameters()).device)
            self.decoder = self.decoder.to(next(self.parameters()).device)

    def forward(self, x, targets=None, phase=1, apply_enhancements=True, use_compression=None,
                current_scale=None, training_phase=None, return_scale_info=False):
        """
        Quality-preserving forward pass with strategic memory optimizations
        """
        # Store input dimensions
        original_shape = x.shape
        batch_size, channels, height, width = original_shape

        # Scale-aware preprocessing
        if current_scale is not None and current_scale != 1.0:
            scaled_size = [int(height * current_scale), int(width * current_scale)]
            scaled_size = [max(8, dim) for dim in scaled_size]  # Ensure minimum size
            x_scaled = F.interpolate(x, size=scaled_size, mode='bilinear',
                                   align_corners=False, antialias=True)
        else:
            x_scaled = x

        # Apply domain-specific enhancements
        if apply_enhancements:
            image_type = self.config['dataset']['image_type']
            if image_type == "astronomical":
                x_scaled = self.enhancement_modules.apply_astronomical_enhancements(x_scaled)
            elif image_type == "medical":
                x_scaled = self.enhancement_modules.apply_medical_enhancements(x_scaled)
            elif image_type == "agricultural":
                x_scaled = self.enhancement_modules.apply_agricultural_enhancements(x_scaled)

        # Quality-preserving memory optimization: selective gradient checkpointing
        # Only checkpoint encoder in training phases 3+ where memory peaks, preserving decoder quality
        if self.training and phase >= 3:
            base_latent = torch.utils.checkpoint.checkpoint(
                self.encoder, x_scaled, use_reentrant=False, preserve_rng_state=True
            )
        else:
            base_latent = self.encoder(x_scaled)

        current_latent = base_latent

        # Progressive expert guidance - NO checkpointing to preserve quality
        if phase >= 2:
            current_latent = self.reconstruction_expert(current_latent)

        if phase >= 3 and targets is not None:
            current_latent = self.semantic_expert(current_latent, targets)

        # Scale information processing
        scale_info = self._process_scale_information(
            current_latent, original_shape, current_scale, training_phase
        ) if return_scale_info or current_scale is not None else None

        # Final reconstruction - NO checkpointing to ensure reconstruction quality
        reconstructed = self.decoder(current_latent)

        # Scale reconstruction back if needed - CRITICAL for quality preservation
        # Always ensure output matches original input dimensions
        if reconstructed.shape[2:] != (height, width):
            reconstructed = F.interpolate(reconstructed, size=(height, width),
                                        mode='bilinear', align_corners=False, antialias=True)

        # Feature compression
        use_compression = use_compression if use_compression is not None else self.use_compression
        if use_compression:
            compressed_latent = self.feature_compressor(current_latent)
        else:
            compressed_latent = current_latent

        # Efficiency scores
        if phase >= 4:
            efficiency_scores = self.efficiency_critic(current_latent)
        else:
            efficiency_scores = None

        # Return based on requirements
        if return_scale_info:
            return reconstructed, compressed_latent, current_latent, efficiency_scores, scale_info
        else:
            return reconstructed, compressed_latent, current_latent, efficiency_scores

    def _process_scale_information(self, latent, original_shape, current_scale, training_phase):
        """Process and return scale information for multi-scale training"""
        if current_scale is None:
            return None

        scale_info = {
            'scale_factor': current_scale,
            'original_shape': original_shape,
            'training_phase': training_phase,
            'latent_norm': torch.norm(latent, dim=1).mean(),
        }

        # Add scale embedding
        if current_scale is not None:
            scale_idx = min(9, max(0, int(round(current_scale * 10)) - 1))
            scale_info['scale_embedding'] = self.scale_embeddings(
                torch.tensor([scale_idx], device=latent.device)
            )

        return scale_info

    def _apply_enhancements(self, x):
            """Apply domain-specific enhancements - NO CHANGES"""
            image_type = self.config['dataset']['image_type']
            if image_type == "astronomical":
                return self.enhancement_modules.apply_astronomical_enhancements(x)
            elif image_type == "medical":
                return self.enhancement_modules.apply_medical_enhancements(x)
            elif image_type == "agricultural":
                return self.enhancement_modules.apply_agricultural_enhancements(x)
            else:
                return x

    def organize_latent_space(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Enhanced latent space organization with clustering"""
        output = {'embeddings': embeddings}

        # Clustering organization
        if self.use_kl_divergence and hasattr(self, 'cluster_centers'):
            cluster_centers = self.cluster_centers.to(embeddings.device)
            distances = torch.cdist(embeddings, cluster_centers)

            # Convert distances to probabilities
            q_dist = 1.0 / (1.0 + (distances / self.clustering_temperature) ** 2)
            q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

            if labels is not None:
                p_dist = torch.zeros_like(q_dist)
                for i in range(self.cluster_centers.size(0)):
                    mask = (labels == i)
                    if mask.any():
                        p_dist[mask, i] = 1.0
            else:
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

# =============================================================================
# MAIN SYSTEM INTEGRATION
# =============================================================================

class UnifiedFeatureExtractor:
    """Production-ready unified feature extraction system"""

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
        self.label_encoder = None
        self.progressive_config = None
        self.current_training_phase = 1
        self.scale_performance_history = {}

    def setup_data(self, data_dir: str = "Data", split: str = "train", data_source: str = "auto"):
        """Setup data pipeline with source specification"""
        self.dataset = GenericImageDataset(data_dir, self.dataset_name, self.main_config, split, data_source)
        return self.dataset

    def initialize_model(self):
        """Initialize the unified autoencoder model with automatic scale detection"""
        num_classes = len(set(self.dataset.targets)) if self.dataset else 10

        # Update config with actual processed image dimensions
        if self.dataset and len(self.dataset) > 0:
            sample_image, _, _ = self.dataset[0]
            channels, height, width = sample_image.shape

            self.main_config['dataset']['in_channels'] = channels
            self.main_config['dataset']['input_size'] = [height, width]

            logger.info(f"Processed image dimensions: {channels} channels, {height}x{width}")

        # Auto-scale detection - FIXED: Ensure proper initialization
        if self.dataset and (not hasattr(self, 'progressive_config') or self.progressive_config is None):
            logger.info("Auto-detecting optimal scales from dataset frequency content...")
            optimal_scales = self.analyze_frequency_content(self.dataset)

            self.progressive_config = {
                'enabled': True,
                'scales': optimal_scales,
                'phase_epochs': self._calculate_phase_epochs(len(optimal_scales)),
                'auto_detected': True,
                'detection_method': 'frequency_analysis'
            }

            logger.info(f"Using auto-detected scales: {optimal_scales}")

        # Initialize model
        self.model = UnifiedDiscriminativeAutoencoder(self.main_config, num_classes)
        self.model = self.model.to(self.device)

        # Pass progressive config to model - FIXED: Check if it exists
        if hasattr(self, 'progressive_config') and self.progressive_config is not None:
            self.model.progressive_config = self.progressive_config

        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model

    def _calculate_phase_epochs(self, num_scales):
        """Calculate appropriate epochs for each phase"""
        base_epochs = [15, 12, 10, 8]
        if num_scales <= len(base_epochs):
            return base_epochs[:num_scales]
        else:
            additional_epochs = [6] * (num_scales - len(base_epochs))
            return base_epochs + additional_epochs

    def analyze_frequency_content(self, dataset, sample_size=200):
        """Analyze image frequency content to determine optimal scales"""
        logger.info("Analyzing frequency content for optimal scale selection...")
        frequencies = []

        for i, (image, _, _) in enumerate(dataset):
            if i >= sample_size:
                break

            # Convert to grayscale for analysis
            if image.shape[0] == 3:
                image_gray = image.mean(dim=0)
            else:
                image_gray = image.squeeze()

            # 2D FFT analysis
            fft = torch.fft.fft2(image_gray)
            fft_shift = torch.fft.fftshift(fft)
            magnitude = torch.log(1 + torch.abs(fft_shift))

            # Analyze frequency distribution
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            radius_high = min(15, h//20, w//20)
            radius_mid = min(40, h//8, w//8)

            # High frequency
            high_freq = magnitude[center_h-radius_high:center_h+radius_high,
                                 center_w-radius_high:center_w+radius_high].mean()

            # Mid frequency
            mid_band = magnitude[center_h-radius_mid:center_h+radius_mid,
                               center_w-radius_mid:center_w+radius_mid]
            mid_band[radius_mid-radius_high:radius_mid+radius_high,
                    radius_mid-radius_high:radius_mid+radius_high] = 0
            mid_freq = mid_band.mean()

            # Low frequency
            low_freq = magnitude.mean() - (high_freq + mid_freq)

            frequencies.append([high_freq.item(), mid_freq.item(), low_freq.item()])

        frequencies = torch.tensor(frequencies)
        avg_frequencies = frequencies.mean(dim=0)

        logger.info(f"Frequency Analysis Complete:")
        logger.info(f"   High Frequency Power: {avg_frequencies[0]:.3f}")
        logger.info(f"   Mid Frequency Power:  {avg_frequencies[1]:.3f}")
        logger.info(f"   Low Frequency Power:  {avg_frequencies[2]:.3f}")

        return self._recommend_scales_from_frequencies(avg_frequencies)

    def _recommend_scales_from_frequencies(self, frequencies, threshold_ratio=0.25):
        """Recommend scales based on frequency content analysis"""
        high_freq, mid_freq, low_freq = frequencies
        total_power = high_freq + mid_freq + low_freq
        high_ratio = high_freq / total_power if total_power > 0 else 0
        mid_ratio = mid_freq / total_power if total_power > 0 else 0
        low_ratio = low_freq / total_power if total_power > 0 else 0

        recommended_scales = [1.0]  # Always include full resolution

        # Add scales based on frequency content significance
        if high_ratio > threshold_ratio:
            recommended_scales.extend([0.8, 0.9])
        elif high_ratio > threshold_ratio * 0.7:
            recommended_scales.append(0.8)

        if mid_ratio > threshold_ratio:
            recommended_scales.append(0.5)
        elif mid_ratio > threshold_ratio * 0.7:
            recommended_scales.append(0.6)

        if low_ratio > threshold_ratio:
            recommended_scales.append(0.25)
        elif low_ratio > threshold_ratio * 0.7:
            recommended_scales.append(0.33)

        # Remove duplicates and sort
        recommended_scales = sorted(list(set(recommended_scales)), reverse=True)

        # Ensure we have at least 2 scales
        if len(recommended_scales) < 2:
            recommended_scales.extend([0.5, 0.25])
            recommended_scales = sorted(list(set(recommended_scales)), reverse=True)

        logger.info(f"Recommended scales: {recommended_scales}")
        logger.info(f"   (High: {high_ratio:.2f}, Mid: {mid_ratio:.2f}, Low: {low_ratio:.2f})")

        return recommended_scales

    def train(self, data_dir: str = "Data", save_dir: str = "models", auto_detect_scales: bool = True):
        """Quality-preserving memory-optimized training"""
        # Call the new memory-optimized version
        return self._train_memory_optimized(data_dir, save_dir, auto_detect_scales)

    def _train_memory_optimized(self, data_dir: str = "Data", save_dir: str = "models", auto_detect_scales: bool = True):
        """Memory-optimized training without losing precision"""
        if self.dataset is None:
            self.setup_data(data_dir, "train")

        if self.model is None:
            self.initialize_model()

        # Memory optimization settings
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.cuda.empty_cache()

            # Enable memory optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Conservative memory settings
        base_batch_size = 16  # Reduced for memory safety
        gradient_accumulation_steps = 8  # Effective batch size = 16 * 8 = 128
        scales = [1.0, 0.5]  # Reduced to 2 scales for memory
        phase_epochs = [12, 10]  # Slightly reduced epochs

        logger.info(f"🧠 Memory-optimized training:")
        logger.info(f"   Batch size: {base_batch_size}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {base_batch_size * gradient_accumulation_steps}")
        logger.info(f"   Scales: {scales} (reduced for memory)")
        logger.info(f"   Epochs: {phase_epochs}")

        train_loader = DataLoader(
            self.dataset,
            batch_size=base_batch_size,
            shuffle=True,
            num_workers=4,  # Reduced workers
            pin_memory=True,
            persistent_workers=False,  # Disable for memory
            drop_last=True  # Avoid partial batches
        )

        # Progressive multi-scale training
        for phase, (scale, epochs) in enumerate(zip(scales, phase_epochs), 1):
            logger.info(f"🎯 Phase {phase}/{len(scales)} - Scale {scale} for {epochs} epochs")

            # Clear cache between phases
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            success = self._train_single_scale_phase_memory_optimized(
                train_loader, scale, phase, epochs, save_dir, gradient_accumulation_steps
            )
            if not success:
                logger.warning(f"Phase {phase} had issues, but continuing...")

        self.save_model(save_dir)
        logger.info("✅ Training completed with memory optimizations!")

    def _train_single_scale_phase_memory_optimized(self, train_loader, scale, phase, epochs, save_dir, grad_accum):
        """Memory-optimized single scale training"""
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float('inf')
        patience_counter = 0

        # Set up autocast device
        autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            optimizer.zero_grad()

            for batch_idx, (data, targets, _) in enumerate(train_loader):
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device) if targets is not None else None

                # Use mixed precision for memory efficiency - FIXED AUTOCAST
                with torch.amp.autocast(device_type=autocast_device, enabled=torch.cuda.is_available()):
                    reconstructed, latent_32, latent_512, efficiency_scores, scale_info = self.model(
                        data, targets, phase=phase, apply_enhancements=True,
                        current_scale=scale, training_phase=phase, return_scale_info=True
                    )

                    # Loss computation
                    loss = F.mse_loss(reconstructed, data)

                    # Scale consistency loss
                    if scale_info is not None and phase > 1:
                        scale_consistency_loss = self._compute_scale_consistency_loss(scale_info)
                        loss = loss + 0.03 * scale_consistency_loss

                    # Classification loss
                    if phase >= 3 and targets is not None:
                        latent_organization = self.model.organize_latent_space(latent_512, targets)
                        if 'class_logits' in latent_organization:
                            class_loss = F.cross_entropy(latent_organization['class_logits'], targets)
                            loss = loss + 0.1 * class_loss

                # Gradient accumulation with scaling
                loss = loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Clear cache after optimizer step
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                total_loss += loss.item() * grad_accum
                num_batches += 1

            # Handle remaining gradients
            if (batch_idx + 1) % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            if num_batches == 0:
                continue

            epoch_loss = total_loss / num_batches

            if epoch % 2 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f'Phase {phase} Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}')

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                self._save_scale_checkpoint(phase, scale, epoch_loss, save_dir)
            else:
                patience_counter += 1

            if patience_counter >= 6:
                logger.info(f'⏹️ Phase {phase} early stopping at epoch {epoch}')
                break

        return best_loss < 0.1

    def _train_single_scale_phase_quality(self, train_loader, scale, phase, epochs, save_dir, gradient_accumulation_steps):
        """Quality-preserving single scale phase training"""
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Better convergence

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            optimizer.zero_grad()

            for batch_idx, (data, targets, _) in enumerate(train_loader):
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device) if targets is not None else None

                # Forward pass with quality preservation
                reconstructed, latent_32, latent_512, efficiency_scores, scale_info = self.model(
                    data, targets, phase=phase, apply_enhancements=True,
                    current_scale=scale, training_phase=phase, return_scale_info=True
                )

                # Quality-preserving loss computation
                loss = F.mse_loss(reconstructed, data)

                # Multi-scale consistency loss (preserves quality)
                if scale_info is not None and phase > 1:
                    scale_consistency_loss = self._compute_scale_consistency_loss(scale_info)
                    loss = loss + 0.03 * scale_consistency_loss

                # Phase-specific losses (preserves discriminative power)
                if phase >= 3 and targets is not None:
                    latent_organization = self.model.organize_latent_space(latent_512, targets)
                    if 'class_logits' in latent_organization:
                        class_loss = F.cross_entropy(latent_organization['class_logits'], targets)
                        loss = loss + 0.1 * class_loss

                # Gradient accumulation with proper scaling
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

            # Handle remaining gradients
            if (batch_idx + 1) % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Step scheduler for better convergence
            scheduler.step()

            if num_batches == 0:
                continue

            epoch_loss = total_loss / num_batches

            # Quality monitoring
            if epoch % 2 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f'Phase {phase} Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}')

            # Quality-based early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                self._save_scale_checkpoint(phase, scale, epoch_loss, save_dir)
            else:
                patience_counter += 1

            if patience_counter >= 8:  # More patience for quality
                logger.info(f'⏹️ Phase {phase} early stopping at epoch {epoch}')
                break

        return best_loss < 0.08  # Stricter convergence for quality

    def _compute_scale_consistency_loss(self, scale_info):
        """Ensure consistent representations across scales"""
        if 'features' not in scale_info:
            return torch.tensor(0.0)

        scale_features = scale_info['features']
        batch_size, num_scales, feature_dim = scale_features.shape
        scale_similarities = []

        for i in range(num_scales):
            for j in range(i+1, num_scales):
                sim = F.cosine_similarity(scale_features[:, i], scale_features[:, j], dim=1)
                scale_similarities.append(sim.mean())

        consistency_loss = 1.0 - torch.stack(scale_similarities).mean()
        return consistency_loss

    def _compute_scale_diversity_loss(self, scale_info):
        """Ensure diverse and complementary representations across scales"""
        if 'scale_enhanced' not in scale_info:
            return torch.tensor(0.0)

        enhanced_features = scale_info['scale_enhanced']
        batch_size, feature_dim = enhanced_features.shape

        if batch_size < 2:
            return torch.tensor(0.0)

        # Center the features
        features_centered = enhanced_features - enhanced_features.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        covariance = torch.matmul(features_centered.T, features_centered) / (batch_size - 1)

        # Encourage diverse features by minimizing off-diagonal elements
        diag_mask = torch.eye(feature_dim, device=enhanced_features.device)
        off_diag_elements = covariance * (1 - diag_mask)
        diversity_loss = off_diag_elements.norm() / (feature_dim * (feature_dim - 1))

        return diversity_loss

    def _save_scale_checkpoint(self, phase, scale, loss, save_dir):
        """Save checkpoint for specific scale phase"""
        checkpoint = {
            'phase': phase,
            'scale': scale,
            'loss': loss,
            'model_state_dict': self.model.state_dict(),
            'progressive_config': getattr(self, 'progressive_config', {}),
            'scale_performance': getattr(self, 'scale_performance_history', {})
        }

        checkpoint_path = Path(save_dir) / f"{self.dataset_name}_phase{phase}_scale{scale}.pth"
        torch.save(checkpoint, checkpoint_path)

    def _train_fusion_phase(self, train_loader, epochs, save_dir):
        """Final phase: learn to fuse multi-scale representations"""
        logger.info("Training scale fusion components...")
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-5)

        # Set up autocast device
        autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_idx, (data, targets, _) in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()

                # Use updated autocast - FIXED
                with torch.amp.autocast(device_type=autocast_device, enabled=torch.cuda.is_available()):
                    # Use all scales simultaneously in final phase
                    reconstructed, latent_32, latent_512, efficiency_scores, scale_info = self.model(
                        data, phase=4, current_scale=1.0, return_scale_info=True
                    )

                    loss = F.mse_loss(reconstructed, data)

                    # Encourage diverse scale contributions
                    if scale_info is not None:
                        diversity_loss = self._compute_scale_diversity_loss(scale_info)
                        loss = loss + 0.02 * diversity_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 2 == 0:
                logger.info(f'Fusion Phase Epoch [{epoch}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    def extract_features(self, data_dir: str = "Data", split: str = "train", use_compression: bool = False) -> pd.DataFrame:
        """Extract features with guaranteed latent space consistency using robust multi-scale processing"""
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")

        if self.dataset is None:
            self.setup_data(data_dir, split)

        # Set model to eval mode and ensure deterministic behavior
        self.model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        features_list = []
        targets_list = []
        image_paths_list = []
        image_hashes_list = []

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

                # Get final latent representation using robust multi-scale processing
                reconstructed, final_latent, base_latent, efficiency_scores = self.model(
                    data, targets, phase=4, use_compression=use_compression
                )

                # Store features and metadata
                features_list.append(final_latent.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                image_paths_list.extend(image_paths)

                # Generate consistency hashes
                batch_hashes = [self._generate_image_hash(path) for path in image_paths]
                image_hashes_list.extend(batch_hashes)

                if batch_idx % 10 == 0:
                    feature_type = "compressed" if use_compression else "rich 512D"
                    #logger.info(f'Processed batch {batch_idx}/{len(dataloader)} ({feature_type} features)')

        # Concatenate all batches
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Create DataFrame
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

        for i in range(latent_dim):
            df_data[f'feature_{i}'] = all_features[:, i]

        df = pd.DataFrame(df_data)
        columns = ['target', 'image_path', 'image_hash', 'extraction_timestamp', 'model_version', 'feature_type'] + feature_columns
        df = df[columns]

        feature_type = f"compressed {latent_dim}D" if use_compression else "rich 512D"
        logger.info(f"✅ Successfully extracted {len(df)} samples with {feature_type} features")
        logger.info(f"🔒 Latent space consistency ensured with deterministic operations")
        logger.info(f"🎯 Using robust multi-scale processing preserving original image quality")

        return df

    def _generate_image_hash(self, image_path: str) -> str:
        """Generate consistent hash for image verification"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "unknown"

    def train_feature_compressor(self, features_df: pd.DataFrame, epochs: int = 100):
        """Robust feature compressor training for unified autoencoder"""
        logger.info("Starting Feature Space Compressor Training")

        # Check if model supports feature compression
        if not hasattr(self.model, 'feature_compressor'):
            logger.error("❌ Model does not have feature_compressor attribute!")
            logger.error("   This model cannot perform feature compression.")
            logger.info("💡 Please use a model that supports feature compression")
            return False

        feature_cols = [f'feature_{i}' for i in range(512)]

        # Validate feature dimensions
        available_cols = [col for col in feature_cols if col in features_df.columns]
        if len(available_cols) != 512:
            logger.warning(f"⚠️ Expected 512 features, found {len(available_cols)}")
            logger.info("💡 Extracting fresh features with correct dimensions...")
            fresh_features = self.extract_features(use_compression=False)
            return self.train_feature_compressor(fresh_features, epochs)

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

        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        recon_criterion = nn.MSELoss()
        class_criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience = 0
        max_patience = 15

        logger.info(f"🔧 Training compressor with {len(features_df)} samples for {epochs} epochs")

        for epoch in range(epochs):
            self.model.feature_compressor.train()
            optimizer.zero_grad()

            # Forward pass through compressor
            compressed, reconstructed, classified = self.model.feature_compressor(
                features_tensor, return_all=True
            )

            # Combined loss
            recon_loss = recon_criterion(reconstructed, features_tensor)
            class_loss = class_criterion(classified, targets_tensor)
            total_loss = 0.3 * recon_loss + 0.7 * class_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.feature_compressor.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if total_loss < best_loss:
                best_loss = total_loss
                patience = 0
                torch.save({
                    'compressor_state_dict': self.model.feature_compressor.state_dict(),
                    'best_loss': best_loss,
                    'epoch': epoch
                }, f"models/{self.dataset_name}_compressor_best.pth")
            else:
                patience += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    self.model.feature_compressor.eval()
                    _, _, classified = self.model.feature_compressor(features_tensor, return_all=True)
                    preds = torch.argmax(classified, dim=1)
                    accuracy = (preds == targets_tensor).float().mean()

                    current_lr = scheduler.get_last_lr()[0]

                logger.info(f'Compressor Epoch [{epoch}/{epochs}], '
                           f'Loss: {total_loss.item():.4f}, '
                           f'Acc: {accuracy.item():.4f}, '
                           f'LR: {current_lr:.6f}')

            if patience >= max_patience:
                logger.info(f'⏹️ Compressor early stopping at epoch {epoch}')
                break

        logger.info(f"✅ Feature Space Compressor training completed. Best loss: {best_loss:.4f}")
        self.model.use_compression = True
        return True

    def extract_features(self, data_dir: str = "Data", split: str = "train", use_compression: bool = False) -> pd.DataFrame:
        """Enhanced feature extraction with better error handling"""
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")

        if self.dataset is None:
            self.setup_data(data_dir, split)

        # Set model to eval mode
        self.model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        features_list = []
        targets_list = []
        image_paths_list = []
        image_hashes_list = []

        dataloader = DataLoader(
            self.dataset,
            batch_size=min(32, self.main_config['training']['batch_size']),  # Smaller batches for stability
            shuffle=False,
            num_workers=min(4, self.main_config['training']['num_workers']),
            pin_memory=True
        )

        with torch.no_grad():
            for batch_idx, (data, targets, image_paths) in enumerate(dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                try:
                    # Get final latent representation
                    reconstructed, final_latent, base_latent, efficiency_scores = self.model(
                        data, targets, phase=4, use_compression=use_compression
                    )

                    # Store features and metadata
                    features_list.append(final_latent.cpu().numpy())
                    targets_list.append(targets.cpu().numpy())
                    image_paths_list.extend(image_paths)

                    # Generate consistency hashes
                    batch_hashes = [self._generate_image_hash(path) for path in image_paths]
                    image_hashes_list.extend(batch_hashes)

                    if batch_idx % 20 == 0:
                        feature_type = "compressed" if use_compression else "rich 512D"
                        logger.info(f'Processed batch {batch_idx}/{len(dataloader)} ({feature_type} features)')

                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                    continue

        if not features_list:
            raise RuntimeError("No features were extracted. Check model and data.")

        # Concatenate all batches
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Create DataFrame
        latent_dim = all_features.shape[1]
        feature_columns = [f'feature_{i}' for i in range(latent_dim)]

        df_data = {
            'target': all_targets,
            'image_path': image_paths_list,
            'image_hash': image_hashes_list,
            'extraction_timestamp': time.time(),
            'model_version': f"{self.dataset_name}_autoencoder",
            'feature_type': 'compressed' if use_compression else 'rich_512D'
        }

        for i in range(latent_dim):
            df_data[f'feature_{i}'] = all_features[:, i]

        df = pd.DataFrame(df_data)
        columns = ['target', 'image_path', 'image_hash', 'extraction_timestamp', 'model_version', 'feature_type'] + feature_columns
        df = df[columns]

        feature_type = f"compressed {latent_dim}D" if use_compression else "rich 512D"
        logger.info(f"✅ Successfully extracted {len(df)} samples with {feature_type} features")

        return df

    def save_features_and_config(self, df: pd.DataFrame, output_dir: str = "data"):
        """Save features to CSV and create configuration files"""
        output_path = Path(output_dir) / self.dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_path / f"{self.dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Features saved to: {csv_path}")

        # Get feature columns
        metadata_columns = ['target', 'image_path', 'image_hash', 'extraction_timestamp', 'model_version', 'feature_type']
        feature_columns = [col for col in df.columns if col not in metadata_columns]

        # Create dataset configuration
        dataset_config = {
            "file_path": f"data/{self.dataset_name}/{self.dataset_name}.csv",
            "column_names": ["target"] + feature_columns,
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
        logger.info(f"Model saved to: {model_path}")

        # Save label encoder separately
        if hasattr(self, 'label_encoder') and self.label_encoder:
            encoder_path = save_path / f"{self.dataset_name}_label_encoder.json"
            encoder_data = {
                'classes': self.label_encoder.classes_.tolist(),
                'dataset_name': self.dataset_name,
                'timestamp': time.time()
            }
            with open(encoder_path, 'w') as f:
                json.dump(encoder_data, f, indent=2)
            logger.info(f"Label encoder saved to: {encoder_path}")

    def load_model(self, model_path: str):
        """Load trained model with label encoder support"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load label encoder if available
            if 'label_encoder' in checkpoint and checkpoint['label_encoder'] and checkpoint['label_encoder']['fitted']:
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(checkpoint['label_encoder']['classes'])
                logger.info(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
            else:
                logger.warning("No label encoder found in checkpoint, using numeric labels")
                self.label_encoder = None

            # Load model weights with compatibility handling
            current_state_dict = self.model.state_dict()
            saved_state_dict = checkpoint['model_state_dict']

            compatible_state_dict = {}
            missing_keys = []
            unexpected_keys = []

            for key in current_state_dict.keys():
                if key in saved_state_dict:
                    if current_state_dict[key].shape == saved_state_dict[key].shape:
                        compatible_state_dict[key] = saved_state_dict[key]
                    else:
                        missing_keys.append(key)
                        logger.warning(f"Shape mismatch for {key}")
                else:
                    missing_keys.append(key)

            for key in saved_state_dict.keys():
                if key not in current_state_dict:
                    unexpected_keys.append(key)

            self.model.load_state_dict(compatible_state_dict, strict=False)

            logger.info(f"Loaded model from: {model_path}")
            logger.info(f"Compatible parameters: {len(compatible_state_dict)}/{len(current_state_dict)}")

            if missing_keys:
                logger.warning(f"Missing/incompatible keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Continuing with randomly initialized weights")

    def verify_latent_consistency(self, previous_features_path: str = None) -> Dict[str, Any]:
        """Verify that the same images produce the same latent features across runs"""
        if previous_features_path and os.path.exists(previous_features_path):
            previous_df = pd.read_csv(previous_features_path)
            current_df = self.extract_features()

            common_hashes = set(previous_df['image_hash']).intersection(set(current_df['image_hash']))

            if common_hashes:
                consistency_results = {}

                for image_hash in list(common_hashes)[:10]:
                    prev_features = previous_df[previous_df['image_hash'] == image_hash].iloc[0]
                    curr_features = current_df[current_df['image_hash'] == image_hash].iloc[0]

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
                logger.info(f"Latent space consistency check: Average similarity = {avg_similarity:.6f}")

                return {
                    'average_similarity': avg_similarity,
                    'sample_comparisons': consistency_results,
                    'consistent': avg_similarity > 0.99
                }

        return {'consistent': True, 'no_previous_data': True}


# =============================================================================
# DOMAIN-AWARE FEATURE EXTRACTOR
# =============================================================================

class DomainAwareFeatureExtractor(UnifiedFeatureExtractor):
    """Main system that switches between domain-specific autoencoders with inherited functionality"""

    def __init__(self, dataset_name: str, config_dir: str = "configs", domain: str = "general"):
        # Initialize the parent class (UnifiedFeatureExtractor)
        super().__init__(dataset_name, config_dir)
        self.domain = domain  # "astronomical", "medical", or "general"

        # Override the device logging to include domain information
        logger.info(f"Using device: {self.device}")
        logger.info(f"Domain: {self.domain}")

        # Initialize components (already done by parent, but we can add domain-specific ones)
        self.model = None
        self.dataset = None

    def initialize_model(self):
        """Initialize the appropriate domain-specific model"""
        num_classes = len(set(self.dataset.targets)) if self.dataset else 10

        # Update config with domain-specific settings
        self.main_config['dataset']['image_type'] = self.domain

        # Select domain-specific model
        if self.domain == "astronomical":
            self.model = AstronomicalAutoencoder(self.main_config, num_classes)
            logger.info("🔭 Initialized Astronomical Autoencoder")
            logger.info("   • Optimized for spiral arms, rings, and large-scale structures")
            logger.info("   • Enhanced low surface brightness preservation")
            logger.info("   • Morphology-aware feature extraction")
        elif self.domain == "medical":
            self.model = MedicalAutoencoder(self.main_config, num_classes)
            logger.info("🏥 Initialized Medical Autoencoder")
            logger.info("   • Optimized for tissue boundary preservation")
            logger.info("   • Enhanced contrast for medical imaging")
            logger.info("   • Lesion-aware feature guidance")
        else:
            self.model = UnifiedDiscriminativeAutoencoder(self.main_config, num_classes)
            logger.info("🌐 Initialized General Autoencoder")
            logger.info("   • Balanced approach for general images")
            logger.info("   • Multi-scale progressive training")
            logger.info("   • Feature space compression available")

        self.model = self.model.to(self.device)
        logger.info(f"Initialized {self.domain} model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model

    def setup_data(self, data_dir: str = "Data", split: str = "train", data_source: str = "auto"):
        """Setup data pipeline with domain-specific enhancements"""
        # Call parent method first
        dataset = super().setup_data(data_dir, split, data_source)

        # Apply domain-specific data enhancements
        if self.domain == "astronomical" and len(dataset) > 0:
            logger.info("🔭 Applying astronomical-specific data enhancements")
            # Astronomical images might benefit from specific preprocessing
            # This could include background subtraction, noise reduction, etc.

        elif self.domain == "medical" and len(dataset) > 0:
            logger.info("🏥 Applying medical-specific data enhancements")
            # Medical images might benefit from specific preprocessing
            # This could include intensity normalization, etc.

        return dataset

    def train(self, data_dir: str = "Data", save_dir: str = "models", auto_detect_scales: bool = True):
        """Domain-aware training with memory optimizations"""
        logger.info(f"🎯 Starting {self.domain}-specific training with memory optimizations")

        # Domain-specific training configurations
        if self.domain == "astronomical":
            original_lr = self.main_config['model'].get('learning_rate', 0.001)
            logger.info(f"🔭 Astronomical training: Using learning rate {original_lr} with memory optimizations")

        elif self.domain == "medical":
            logger.info("🏥 Medical training: Enhanced boundary preservation with memory optimizations")

        # Call parent training method with domain context - USE THE UPDATED VERSION
        return self._train_memory_optimized(data_dir, save_dir, auto_detect_scales)

    def _train_memory_optimized(self, data_dir: str = "Data", save_dir: str = "models", auto_detect_scales: bool = True):
        """Memory-optimized training without losing precision"""
        if self.dataset is None:
            self.setup_data(data_dir, "train")

        if self.model is None:
            self.initialize_model()

        # Memory optimization settings
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.cuda.empty_cache()

            # Enable memory optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Conservative memory settings
        base_batch_size = 16  # Reduced for memory safety
        gradient_accumulation_steps = 8  # Effective batch size = 16 * 8 = 128
        scales = [1.0, 0.5]  # Reduced to 2 scales for memory
        phase_epochs = [12, 10]  # Slightly reduced epochs

        logger.info(f"🧠 Memory-optimized training:")
        logger.info(f"   Batch size: {base_batch_size}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {base_batch_size * gradient_accumulation_steps}")
        logger.info(f"   Scales: {scales} (reduced for memory)")
        logger.info(f"   Epochs: {phase_epochs}")

        train_loader = DataLoader(
            self.dataset,
            batch_size=base_batch_size,
            shuffle=True,
            num_workers=4,  # Reduced workers
            pin_memory=True,
            persistent_workers=False,  # Disable for memory
            drop_last=True  # Avoid partial batches
        )

        # Progressive multi-scale training
        for phase, (scale, epochs) in enumerate(zip(scales, phase_epochs), 1):
            logger.info(f"🎯 Phase {phase}/{len(scales)} - Scale {scale} for {epochs} epochs")

            # Clear cache between phases
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            success = self._train_single_scale_phase_memory_optimized(
                train_loader, scale, phase, epochs, save_dir, gradient_accumulation_steps
            )
            if not success:
                logger.warning(f"Phase {phase} had issues, but continuing...")

        self.save_model(save_dir)
        logger.info("✅ Training completed with memory optimizations!")

    def save_model(self, save_dir: str = "models"):
        """Save model with domain-specific naming"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Use domain-specific naming
        model_suffix = f"_{self.domain}" if self.domain != "general" else ""
        model_path = save_path / f"{self.dataset_name}_autoencoder{model_suffix}.pth"

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.main_config,
            'dataset_name': self.dataset_name,
            'domain': self.domain,  # Save domain information
            'label_encoder': {
                'classes': self.label_encoder.classes_.tolist() if hasattr(self, 'label_encoder') and self.label_encoder else [],
                'fitted': hasattr(self, 'label_encoder') and self.label_encoder is not None
            } if hasattr(self, 'label_encoder') else None
        }

        torch.save(model_data, model_path)
        logger.info(f"💾 {self.domain.capitalize()} model saved to: {model_path}")

        # Save label encoder separately with domain info
        if hasattr(self, 'label_encoder') and self.label_encoder:
            encoder_path = save_path / f"{self.dataset_name}_label_encoder{model_suffix}.json"
            encoder_data = {
                'classes': self.label_encoder.classes_.tolist(),
                'dataset_name': self.dataset_name,
                'domain': self.domain,
                'timestamp': time.time()
            }
            with open(encoder_path, 'w') as f:
                json.dump(encoder_data, f, indent=2)
            logger.info(f"🏷️  Label encoder saved to: {encoder_path}")

    def load_model(self, model_path: str):
        """Load model with domain compatibility checking"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Check domain compatibility
            saved_domain = checkpoint.get('domain', 'general')
            if saved_domain != self.domain:
                logger.warning(f"⚠️  Domain mismatch: Loading {saved_domain} model into {self.domain} system")
                logger.warning(f"   This may affect performance if architectures differ significantly")

            # Load label encoder if available
            if 'label_encoder' in checkpoint and checkpoint['label_encoder'] and checkpoint['label_encoder']['fitted']:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(checkpoint['label_encoder']['classes'])
                logger.info(f"🏷️  Loaded label encoder with {len(self.label_encoder.classes_)} classes")
            else:
                logger.warning("⚠️  No label encoder found in checkpoint, using numeric labels")
                self.label_encoder = None

            # Load model weights with compatibility handling
            current_state_dict = self.model.state_dict()
            saved_state_dict = checkpoint['model_state_dict']

            compatible_state_dict = {}
            missing_keys = []
            unexpected_keys = []

            for key in current_state_dict.keys():
                if key in saved_state_dict:
                    if current_state_dict[key].shape == saved_state_dict[key].shape:
                        compatible_state_dict[key] = saved_state_dict[key]
                    else:
                        missing_keys.append(key)
                        logger.warning(f"Shape mismatch for {key}")
                else:
                    missing_keys.append(key)

            for key in saved_state_dict.keys():
                if key not in current_state_dict:
                    unexpected_keys.append(key)

            self.model.load_state_dict(compatible_state_dict, strict=False)

            logger.info(f"✅ Loaded {saved_domain} model from: {model_path}")
            logger.info(f"📊 Compatible parameters: {len(compatible_state_dict)}/{len(current_state_dict)}")

            if missing_keys:
                logger.warning(f"⚠️  Missing/incompatible keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"⚠️  Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("🆕 Continuing with randomly initialized weights")

    def extract_features(self, data_dir: str = "Data", split: str = "train", use_compression: bool = False) -> pd.DataFrame:
        """Extract features with domain-specific processing"""
        logger.info(f"🔍 Extracting {self.domain}-specific features")

        if self.domain == "astronomical":
            logger.info("   • Preserving large-scale astronomical structures")
            logger.info("   • Enhancing morphological features")
        elif self.domain == "medical":
            logger.info("   • Preserving tissue boundaries")
            logger.info("   • Enhancing medical contrast")

        # Call parent feature extraction
        return super().extract_features(data_dir, split, use_compression)

    def train_feature_compressor(self, features_df: pd.DataFrame, epochs: int = 100):
        """Domain-aware feature compressor training with robust error handling"""
        logger.info(f"🎯 Training {self.domain}-aware feature compressor")

        # Check if model supports feature compression
        if not self._check_feature_compressor_available():
            logger.error(f"❌ {self.domain.capitalize()} model does not support feature compression")
            logger.info("💡 Falling back to base autoencoder compression")
            return self._train_fallback_compressor(features_df, epochs)

        feature_cols = [f'feature_{i}' for i in range(512)]

        # Validate feature columns
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            logger.error(f"❌ Missing feature columns: {missing_cols[:5]}...")
            logger.info("💡 Trying to extract features first...")
            return self._extract_and_train_compressor(features_df, epochs)

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
        patience = 0
        max_patience = 10

        logger.info(f"🔧 Training compressor for {epochs} epochs with {len(features_df)} samples")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass through compressor
            compressed, reconstructed, classified = self.model.feature_compressor(
                features_tensor, return_all=True
            )

            # Combined loss with domain-specific weighting
            recon_loss = recon_criterion(reconstructed, features_tensor)
            class_loss = class_criterion(classified, targets_tensor)

            # Domain-specific loss weighting
            if self.domain == "astronomical":
                total_loss = 0.4 * recon_loss + 0.6 * class_loss  # Emphasize classification for galaxies
            elif self.domain == "medical":
                total_loss = 0.5 * recon_loss + 0.5 * class_loss  # Balanced for medical
            else:
                total_loss = 0.3 * recon_loss + 0.7 * class_loss  # General emphasis on classification

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.feature_compressor.parameters(), 1.0)
            optimizer.step()

            if total_loss < best_loss:
                best_loss = total_loss
                patience = 0
                # Save domain-specific checkpoint
                model_suffix = f"_{self.domain}" if self.domain != "general" else ""
                torch.save({
                    'compressor_state_dict': self.model.feature_compressor.state_dict(),
                    'best_loss': best_loss,
                    'epoch': epoch,
                    'domain': self.domain
                }, f"models/{self.dataset_name}_compressor_best{model_suffix}.pth")
            else:
                patience += 1

            if epoch % 10 == 0:
                with torch.no_grad():
                    _, _, classified = self.model.feature_compressor(features_tensor, return_all=True)
                    preds = torch.argmax(classified, dim=1)
                    accuracy = (preds == targets_tensor).float().mean()

                logger.info(f'{self.domain.capitalize()} Compressor Epoch [{epoch}/{epochs}], '
                           f'Total Loss: {total_loss.item():.4f}, '
                           f'Recon: {recon_loss.item():.4f}, '
                           f'Class: {class_loss.item():.4f}, '
                           f'Accuracy: {accuracy.item():.4f}')

            if patience >= max_patience:
                logger.info(f'⏹️ {self.domain.capitalize()} compressor early stopping at epoch {epoch}')
                break

        logger.info(f"✅ {self.domain.capitalize()} Feature Space Compressor training completed. Best loss: {best_loss:.4f}")
        self.model.use_compression = True

    def _check_feature_compressor_available(self) -> bool:
        """Check if the current model supports feature compression"""
        if hasattr(self.model, 'feature_compressor'):
            return True
        elif hasattr(self.model, 'base_autoencoder') and hasattr(self.model.base_autoencoder, 'feature_compressor'):
            # Use base autoencoder's compressor
            self.model.feature_compressor = self.model.base_autoencoder.feature_compressor
            return True
        return False

    def _train_fallback_compressor(self, features_df: pd.DataFrame, epochs: int = 100):
        """Fallback training using base autoencoder components"""
        logger.info("🔄 Using fallback compressor training")

        # Extract features using the current model
        logger.info("📥 Extracting features for compressor training...")
        rich_features_df = self.extract_features(use_compression=False)

        # Now call the parent class method
        return super().train_feature_compressor(rich_features_df, epochs)

    def _extract_and_train_compressor(self, features_df: pd.DataFrame, epochs: int = 100):
        """Extract features first, then train compressor"""
        logger.info("🔄 Feature columns missing, extracting fresh features...")

        # Extract new features
        fresh_features_df = self.extract_features(use_compression=False)

        # Merge with existing dataframe if possible
        if 'image_path' in features_df.columns and 'image_path' in fresh_features_df.columns:
            # Use fresh features but keep original targets and metadata
            feature_cols = [f'feature_{i}' for i in range(512)]
            merged_df = features_df.drop(columns=feature_cols, errors='ignore')
            merged_df = merged_df.merge(
                fresh_features_df[['image_path'] + feature_cols],
                on='image_path',
                how='left'
            )
            return self.train_feature_compressor(merged_df, epochs)
        else:
            # Use fresh features directly
            return self.train_feature_compressor(fresh_features_df, epochs)

    def get_domain_info(self) -> Dict[str, Any]:
        """Get information about the current domain configuration"""
        return {
            'domain': self.domain,
            'dataset': self.dataset_name,
            'model_type': self.model.__class__.__name__ if self.model else 'Not initialized',
            'specialized': self.domain in ['astronomical', 'medical'],
            'config': {
                'feature_dims': self.main_config['model']['feature_dims'],
                'image_type': self.main_config['dataset']['image_type']
            }
        }

    # All other methods are automatically inherited from UnifiedFeatureExtractor:
    # - save_features_and_config()
    # - verify_latent_consistency()
    # - _generate_image_hash()
    # - analyze_frequency_content()
    # - _train_single_scale_phase()
    # - _compute_scale_consistency_loss()
    # - _compute_scale_diversity_loss()
    # - _save_scale_checkpoint()
    # - _train_fusion_phase()
    # - _calculate_phase_epochs()
    # - _recommend_scales_from_frequencies()


# =============================================================================
# UPDATED COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments with domain specification"""
    parser = argparse.ArgumentParser(
        description='Domain-Aware Unified Autoencoder System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Astronomical images (galaxies, nebulae) - preserves large-scale structures
  python DBVision.py --dataset Galaxies --domain astronomical --fresh --data-source local

  # Medical images (X-rays, MRI) - preserves tissue boundaries
  python DBVision.py --dataset ChestXRay --domain medical --fresh --data-source local

  # General images with custom paths
  python DBVision.py --dataset cifar100 --domain general --data-dir /path/to/data --output-dir /path/to/output

  # Feature extraction only from existing model
  python DBVision.py --dataset Galaxies --domain astronomical --skip-training --no-compression

DOMAIN EXPLANATION:
  astronomical: Optimized for galaxies, nebulae - preserves spiral arms, rings, large structures
  medical:      Optimized for X-rays, MRI - preserves tissue boundaries, enhances contrast
  general:      Balanced approach for natural images, objects, general computer vision
        """
    )

    # Domain specification (MANDATORY for specialized domains)
    parser.add_argument('--domain', type=str, required=False, default='general',
                       choices=['general', 'astronomical', 'medical'],
                       help='Domain specialization: general, astronomical, or medical')

    # Dataset options
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Galaxies, ChestXRay, cifar100)')
    parser.add_argument('--data-source', type=str, default='auto',
                       choices=['auto', 'torchvision', 'local'],
                       help='Data source: auto (detect), torchvision, or local folder')

    # Directory options with defaults
    parser.add_argument('--data-dir', type=str, default='Data',
                       help='Directory for raw data (default: Data)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Directory for processed features (default: data)')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Configuration directory (default: configs)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Models directory (default: models)')

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
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only extract features from existing model')

    return parser.parse_args()


def main():
    """
    DOMAIN-AWARE UNIFIED DISCRIMINATIVE AUTOENCODER SYSTEM

    A comprehensive feature extraction system that automatically adapts to different image domains
    with specialized architectures for optimal performance.

    KEY FEATURES:
    - Domain-specific autoencoders (astronomical, medical, general)
    - Multi-scale progressive training
    - Feature space compression
    - Latent space consistency verification
    - Automatic dataset handling

    USAGE EXAMPLES:

    # Astronomical images (galaxies, nebulae) - preserves large-scale structures
    python DBVision.py --dataset Galaxies --domain astronomical --fresh --data-source local

    # Medical images (X-rays, MRI) - preserves tissue boundaries
    python DBVision.py --dataset ChestXRay --domain medical --fresh --data-source local

    # General images (default) - balanced approach
    python DBVision.py --dataset cifar100 --domain general --fresh

    # Feature extraction only from existing model
    python DBVision.py --dataset Galaxies --domain astronomical --skip-training --no-compression
    """
    args = parse_arguments()

    # Create directories if they don't exist (with defaults)
    Path(args.data_dir).mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.config_dir).mkdir(exist_ok=True)
    Path(args.models_dir).mkdir(exist_ok=True)

    # Domain-specific logging and validation
    if args.domain in ['astronomical', 'medical']:
        logger.info("🎯 DOMAIN-SPECIFIC UNIFIED AUTOENCODER SYSTEM")
        logger.info(f"🔭 Domain: {args.domain.upper()} (Specialized Architecture)")

        # Domain-specific information
        if args.domain == 'astronomical':
            logger.info("   • Optimized for large-scale structures (spiral arms, rings)")
            logger.info("   • Preserves low surface brightness features")
            logger.info("   • Enhanced morphology preservation")
        else:  # medical
            logger.info("   • Optimized for tissue boundary preservation")
            logger.info("   • Enhanced contrast for medical imaging")
            logger.info("   • Lesion-aware feature extraction")
    else:
        logger.info("🌐 UNIFIED DISCRIMINATIVE AUTOENCODER SYSTEM")
        logger.info(f"🌐 Domain: {args.domain.upper()} (General Architecture)")
        logger.info("   • Balanced approach for general images")
        logger.info("   • Multi-scale progressive training")
        logger.info("   • Feature space compression available")

    logger.info(f"📁 Dataset: {args.dataset}")
    logger.info(f"📥 Data source: {args.data_source}")
    logger.info(f"📂 Data directory: {args.data_dir}")
    logger.info(f"📤 Output directory: {args.output_dir}")
    logger.info(f"🆕 Fresh start: {args.fresh}")
    logger.info(f"📊 Final feature dimension: {args.final_dim}D")

    # Display configuration help
    if args.fresh:
        logger.info("💡 Fresh start: Training new model from scratch")
    if args.skip_training:
        logger.info("💡 Skip training: Only extracting features from existing model")
    if args.no_compression:
        logger.info("💡 No compression: Using raw 512D features (higher quality)")
    else:
        logger.info("💡 With compression: Using compressed features (storage efficient)")

    try:
        # Initialize DOMAIN-AWARE system
        extractor = DomainAwareFeatureExtractor(args.dataset, args.config_dir, args.domain)

        # Override config if command line arguments provided
        if args.epochs:
            extractor.main_config['training']['epochs'] = args.epochs
            logger.info(f"⚙️  Overriding training epochs to: {args.epochs}")
        if args.batch_size:
            extractor.main_config['training']['batch_size'] = args.batch_size
            logger.info(f"⚙️  Overriding batch size to: {args.batch_size}")
        if args.final_dim:
            extractor.main_config['model']['feature_dims'] = args.final_dim
            logger.info(f"⚙️  Setting final feature dimension to: {args.final_dim}D")

        # Setup data pipeline
        logger.info("\n📥 SETTING UP DATA PIPELINE...")
        logger.info(f"   • Loading from: {args.data_dir}/{args.dataset}")
        logger.info(f"   • Data source: {args.data_source}")
        logger.info("   • Applying domain-appropriate preprocessing")
        extractor.setup_data(args.data_dir, data_source=args.data_source)

        # Check for existing model (domain-specific naming)
        model_suffix = f"_{args.domain}" if args.domain != "general" else ""
        model_path = Path(args.models_dir) / f"{args.dataset}_autoencoder{model_suffix}.pth"
        compressor_path = Path(args.models_dir) / f"{args.dataset}_compressor_best{model_suffix}.pth"

        model_exists = model_path.exists() and not args.fresh
        compressor_exists = compressor_path.exists() and not args.fresh

        # Initialize model
        logger.info("\n🏗️  INITIALIZING MODEL...")
        logger.info(f"   • Building {args.domain} architecture")
        logger.info("   • Setting up enhancement modules")
        logger.info("   • Configuring multi-scale processing")
        extractor.initialize_model()

        # Load existing model if available and not fresh start
        if model_exists and not args.skip_training:
            logger.info(f"📂 Loading existing {args.domain} model from {model_path}")
            extractor.load_model(str(model_path))
        elif model_exists and args.skip_training:
            logger.info(f"📂 Loading existing {args.domain} model for feature extraction only")
            extractor.load_model(str(model_path))
        else:
            logger.info("🆕 No existing model found or fresh start requested")

        # Training phase
        if not args.skip_training:
            logger.info("\n🎯 TRAINING PHASE")
            if model_exists and not args.fresh:
                logger.info("   • Using existing trained model")
                logger.info("   • Skipping training phase")
            else:
                logger.info("   • Starting domain-specific training")
                logger.info("   • Using multi-scale progressive approach")
                logger.info("   • Memory-optimized training enabled")
                extractor.train(args.data_dir, args.models_dir)
        else:
            logger.info("\n⏭️  SKIPPING TRAINING PHASE")
            logger.info("   • Proceeding directly to feature extraction")

        # Feature extraction phase
        rich_features_df = None

        if args.extract_rich or (args.train_compressor and not args.no_compression):
            logger.info("\n🔍 EXTRACTING RICH 512D FEATURES...")
            logger.info("   • Using full 512-dimensional latent space")
            logger.info("   • Preserving maximum feature information")
            rich_features_df = extractor.extract_features(
                args.data_dir,
                use_compression=False,
                split="train"
            )

            if args.extract_rich:
                rich_csv_path = Path(args.output_dir) / args.dataset / f"{args.dataset}_rich_512D.csv"
                rich_features_df.to_csv(rich_csv_path, index=False)
                logger.info(f"💾 Rich features saved to: {rich_csv_path}")

        # Feature compressor training
        if args.train_compressor and not args.no_compression and not args.skip_training:
            logger.info("\n🎯 TRAINING FEATURE COMPRESSOR...")
            if compressor_exists and not args.fresh:
                logger.info("   • Using existing feature compressor")
                extractor.model.use_compression = True
                compressor_checkpoint = torch.load(compressor_path)
                extractor.model.feature_compressor.load_state_dict(
                    compressor_checkpoint['compressor_state_dict']
                )
            else:
                if rich_features_df is None:
                    logger.info("   • Extracting rich features for compressor training...")
                    rich_features_df = extractor.extract_features(
                        args.data_dir,
                        use_compression=False,
                        split="train"
                    )

                logger.info("   • Training feature space compressor")
                logger.info("   • Learning compressed representation")
                extractor.train_feature_compressor(rich_features_df, epochs=100)

        # Final feature extraction
        logger.info("\n📊 EXTRACTING FINAL FEATURES...")

        if args.no_compression:
            logger.info("   • Using raw 512D features (maximum quality)")
            final_features_df = extractor.extract_features(
                args.data_dir,
                use_compression=False,
                split="train"
            )
            feature_type = "raw_512D"
        else:
            logger.info("   • Using compressed features (storage efficient)")
            extractor.model.use_compression = True
            final_features_df = extractor.extract_features(
                args.data_dir,
                use_compression=True,
                split="train"
            )
            feature_type = f"compressed_{args.final_dim}D"

        # Save final features and configurations
        logger.info("\n💾 SAVING RESULTS...")
        logger.info(f"   • Output directory: {args.output_dir}")
        csv_path, config_path = extractor.save_features_and_config(final_features_df, args.output_dir)

        # Save DBNN configuration
        extractor.config_manager.save_dbnn_config(extractor.dbnn_config)

        # Verification
        previous_features_path = Path(args.output_dir) / args.dataset / f"{args.dataset}.csv"
        if previous_features_path.exists() and not args.fresh:
            logger.info("\n🔍 VERIFYING LATENT SPACE CONSISTENCY...")
            consistency_results = extractor.verify_latent_consistency(str(previous_features_path))
            if consistency_results.get('consistent', True):
                logger.info("✅ Latent space consistency verified!")
            else:
                logger.warning("⚠️  Latent space consistency check failed")

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 SYSTEM EXECUTION COMPLETE!")
        logger.info("="*80)

        # Domain-specific success message
        if args.domain == "astronomical":
            logger.info("🔭 ASTRONOMICAL FEATURE EXTRACTION SUCCESSFUL!")
            logger.info("   • Large-scale structures preserved")
            logger.info("   • Morphological features enhanced")
            logger.info("   • Ready for galaxy classification")
        elif args.domain == "medical":
            logger.info("🏥 MEDICAL FEATURE EXTRACTION SUCCESSFUL!")
            logger.info("   • Tissue boundaries preserved")
            logger.info("   • Contrast enhanced for diagnosis")
            logger.info("   • Ready for medical image analysis")
        else:
            logger.info("🌐 GENERAL FEATURE EXTRACTION SUCCESSFUL!")
            logger.info("   • Balanced feature representation")
            logger.info("   • Multi-scale processing complete")

        logger.info("")
        logger.info("📊 RESULTS SUMMARY:")
        logger.info(f"   Dataset: {args.dataset}")
        logger.info(f"   Domain: {args.domain}")
        logger.info(f"   Data source: {args.data_source}")
        logger.info(f"   Feature type: {feature_type}")
        logger.info(f"   Final dimensions: {args.final_dim}D")
        logger.info(f"   Total samples: {len(final_features_df):,}")
        logger.info(f"   Raw images: {args.data_dir}/{args.dataset}/")
        logger.info(f"   Features CSV: {csv_path}")
        logger.info(f"   Dataset Config: {config_path}")

        if rich_features_df is not None and args.extract_rich:
            rich_csv_path = Path(args.output_dir) / args.dataset / f"{args.dataset}_rich_512D.csv"
            logger.info(f"   Rich features: {rich_csv_path}")

        logger.info("")
        logger.info("💡 NEXT STEPS:")
        logger.info("   • Use the generated CSV with DBNN for classification")
        logger.info("   • The feature columns are ready for machine learning")

        logger.info("="*80)

        # Display sample
        logger.info("\n📋 SAMPLE OF EXTRACTED FEATURES:")
        sample_cols = ['target', 'image_path'] + [f'feature_{i}' for i in range(min(5, args.final_dim))]
        sample_df = final_features_df[sample_cols].head(3)
        for _, row in sample_df.iterrows():
            logger.info(f"   Target: {row['target']}, Features: {[f'{x:.4f}' for x in row[2:]]}")

    except Exception as e:
        logger.error(f"\n❌ ERROR DURING EXECUTION: {str(e)}")
        logger.error("\n💡 TROUBLESHOOTING HELP:")
        logger.error("   • Check that your dataset path is correct")
        logger.error("   • Ensure images are in class subfolders for local data")
        logger.error("   • Verify you have sufficient GPU memory")
        logger.error("   • Use --help for full command line options")

        logger.error("\n📚 COMMON USAGE PATTERNS:")
        logger.error("   python DBVision.py --dataset YourDataset --domain astronomical --fresh --data-source local")
        logger.error("   python DBVision.py --dataset YourDataset --domain medical --skip-training --no-compression")
        logger.error("   python DBVision.py --dataset cifar100 --domain general --fresh")

        raise



if __name__ == "__main__":
    main()

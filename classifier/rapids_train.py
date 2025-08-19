# Import libraries
import csv
import gc
import os
import tarfile
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image

# Import timm and set cache
os.environ['HF_HOME'] = '/content/hf_cache'
from classifier.py import RiverClassifier

results_dir = "model_performance/"
label_csv_path = "rapids_labels_al_gt_masked.csv"
rapids_images_path = "rapids_label_images_final.tar"

train_version = "masked_al"

class RapidsDataset(Dataset):
    def __init__(self, tar_path, keys, labels, transform=None):
        """
        tar_path: path to your tar file with images
        keys: list of filenames in the tar matching your CSV keys (with extensions)
        labels: list or array of labels aligned with keys
        transform: optional torchvision transforms to apply to images
        """
        self.tar_path = tar_path
        self.keys = keys
        self.labels = labels
        self.transform = transform

        # Open tar once to build the index of members for fast lookup
        with tarfile.open(self.tar_path, "r") as tar:
            self.member_dict = {m.name: m for m in tar.getmembers()}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        key_ext = key + ".jpg"

        # Open tarfile inside __getitem__ for multiprocessing safety
        with tarfile.open(self.tar_path, "r") as tar:
            member = self.member_dict[key_ext]
            file = tar.extractfile(member)
            image = Image.open(file).convert("RGB")
            if self.transform:
                image = self.transform(image)

        label = self.labels[idx]
        return image, label, key

@dataclass
class ModelConfig:
    model_name: str
    hidden_layers: Tuple[int, ...]
    dropout: float
    num_classes: int

@dataclass
class OptimConfig:
    backbone_lr: float
    classifier_lr: float
    weight_decay: float
    scheduler_patience: int
    scheduler_factor: float
    # scheduler variable

# Read CSV containing labels
rapids_df = pd.read_csv(label_csv_path)
key_field = "image"
label_field = "rapid_class"
split_field = "rapid_split"

# The model can be trained on the initial dataset, the
# initial dataset with the addition of masked images for a subset of the
# training data, the initial dataset with the addition of labels added
# through active learning, or the initial dataset with both of the
# above augmentations

if (train_version == "masked"):
    # Remove active learning images
    rapids_df = rapids_df[(rapids_df["al"] == 0)]

    # Keep masked images in the training dataset
    rapids_df = rapids_df[(rapids_df["masked"] == 0) | ((rapids_df["masked"] == 1) & (rapids_df[split_field] == "train"))]

elif (train_version == "al"):
    # Keep active learning but remove masked images
    rapids_df = rapids_df[(rapids_df["masked"] == 0)]

elif (train_version == "masked_al"):
    # Keep masked images in the training dataset
    rapids_df = rapids_df[(rapids_df["masked"] == 0) | ((rapids_df["masked"] == 1) & (rapids_df[split_field] == "train"))]

else:
    # Remove both active learning images and masked images
    rapids_df = rapids_df[(rapids_df["al"] == 0)]
    rapids_df = rapids_df[(rapids_df["masked"] == 0)]

rapids_df = rapids_df.reset_index(drop=True)

# Get train, validation, and test dataset indexes and labels
train_idxs = rapids_df.index[rapids_df[split_field] == "train"].to_numpy()
val_idxs = rapids_df.index[rapids_df[split_field] == "val"].to_numpy()
test_idxs = rapids_df.index[rapids_df[split_field] == "test"].to_numpy()

train_keys = rapids_df[key_field].iloc[train_idxs].to_numpy()
val_keys = rapids_df[key_field].iloc[val_idxs].to_numpy()
test_keys = rapids_df[key_field].iloc[test_idxs].to_numpy()

train_labels = rapids_df[label_field].iloc[train_idxs].to_numpy()
val_labels = rapids_df[label_field].iloc[val_idxs].to_numpy()
test_labels = rapids_df[label_field].iloc[test_idxs].to_numpy()

# Specify model type and get image size
model_name = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k"

classifier_config = ModelConfig(model_name=model_name,
                                hidden_layers=(1024, 512),
                                dropout=0.5,
                                num_classes=2)

# Initiate, train, and evaluate model
classifier = RiverClassifier(classifier_config, model_log_name="resnetv2_al")

image_size = 480 # list(classifier.model.backbone.default_cfg.get("input_size"))[1]
transform_mean = list(classifier.model.backbone.default_cfg.get("mean"))
transform_std = list(classifier.model.backbone.default_cfg.get("std"))

# Define transform now that we have the image size
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std)
])

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std)
])

# Create Datasets and DataLoaders
train_dataset = RapidsDataset(rapids_images_path, train_keys, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

val_dataset = RapidsDataset(rapids_images_path, val_keys, val_labels, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

test_dataset = RapidsDataset(rapids_images_path, test_keys, test_labels, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# ResNetv2
# Unfreeze only the final stage of the backbone and final norm
for param in classifier.model.backbone.stages[3].parameters():
    param.requires_grad = True
for param in classifier.model.backbone.norm.parameters():
    param.requires_grad = True

# (Optional) Confirm classifier is trainable
for param in classifier.model.classifier.parameters():
    param.requires_grad = True

# Initialize the optimizer
optim_config = OptimConfig(backbone_lr = 1e-5, classifier_lr = 1e-3, weight_decay = 1e-5, scheduler_patience = 15, scheduler_factor = 0.1)
classifier.initialize_optimizer(optim_config)

# Train the classifier model
classifier.train(train_loader, val_loader, epochs=5, early_stop_patience=10)

# Evaluate the model on the test data
classifier.evaluate(test_loader)

rapids_df_2 = pd.read_csv(label_csv_path)

rapids_df_2 = rapids_df_2[(rapids_df_2["masked"] == 1) & (rapids_df_2[split_field] == "test")]

rapids_df_2 = rapids_df_2.reset_index(drop=True)

masked_test_idxs = rapids_df_2.index[rapids_df_2[split_field] == "test"].to_numpy()

masked_test_keys = rapids_df_2[key_field].iloc[masked_test_idxs].to_numpy()

masked_test_labels = rapids_df_2[label_field].iloc[masked_test_idxs].to_numpy()

masked_test_dataset = RapidsDataset(rapids_images_path, masked_test_keys, masked_test_labels, transform)
masked_test_loader = DataLoader(masked_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

classifier.evaluate(masked_test_loader)

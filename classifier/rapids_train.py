"""
Train a Rapids Detection Model

Author name and contact: Nicholas Brimhall, ORCID: 0009-0008-7410-0166

This file contains code to train a RiverClassifier model (from the classifier.py file) to predict
the presence of rapids in satellite imagery of rivers. A custom RapidsDataset class is implemented
as a subclass of torch.Dataset to load the images for training. Output from training, including
the final model weights, performance metrics of the model on test data, and plots of the
training and validation loss over time are written to the directory specified by the
artifact_dir variable below. This code and the associated model class were designed to work with
a pretrained ResNetv2_152 model (specified below) loaded from the timm library, but there is some
flexibility in implementing other model types, including other pure convolutional neural network (CNN)
models such as EfficientNetv2, Vision-informed Transformer (ViT) models, or hybrid model such as CoAtNet
provided by timm. If this is done, the user will need to modify the code in the main method of
this script unfreezing the final layers of the pretrained model to work with the backbone
structure of the selected model (or simply remove these lines altogether, although this may
reduce the model's ability to detect rapids).

Dependencies:
    - os, tarfile, dataclasses, pathlib, typing, pandas, PIL
    - torch.utils.data, torchvision.transforms
"""

# Before running this script, the file paths below should be set to the actual locations of the files
# on the user's machine. The rapids_label_dataset.tar file containing the training images, and 
# the rapids_labels.csv file containing the labels, should be downloaded from the CIRRUS data release. 
# Note that the tar file should NOT be extracted/unzipped. 

# Import libraries
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image

# Set timm cache
os.environ['HF_HOME'] = os.path.join(Path.home(), "hf_cache")
from classifier import RiverClassifier

# Set the paths below to their respective locations
# Configuration --------------------------
# Path to tar file containing labeled rapids images
image_dir = os.path.join(Path.home(), "rapids", "rapids_label_dataset.tar")
# Directory where output CSV will be written
artifact_dir = os.path.join(Path.home(), "rapids", "model_performance")
# Path to CSV containing rapid class labels
label_csv_path = os.path.join(Path.home(), "rapids", "rapids_labels.csv")
# File name used for output files
model_log_name = "resnetv2_152_base"
# Model name
model_name = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k"

# train_subset determines which data augmentations are used to train the model. 
# "masked": Retain the ground truth masks in the training dataset
# "al": Retaub the images labeled through active learning in the training dataset
# "masked_al": Retain both the ground truth masks and the images labeled through
# active learning in the training dataset
# else: Retain neither of the above augmentations
train_subset = "base"

class RapidsDataset(Dataset):
    # tar_path: Path to tar file with river images
    # keys: List of file names matching river images in the tar 
    # file to be included in the Dataset
    # labels: List or array of rapid class labels aligned with keys
    # transform: Optional torchvision transforms to apply to images
    def __init__(self, tar_path, keys, labels, transform=None):

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

def main():
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

    if (train_subset == "masked" and "al" in rapids_df.columns):
        # Remove active learning images
        rapids_df = rapids_df[(rapids_df["al"] == 0)]

    elif (train_subset == "al" and "masked" in rapids_df.columns):
        # Keep active learning but remove masked images
        rapids_df = rapids_df[(rapids_df["masked"] == 0)]

    elif (train_subset == "masked_al"):
        # Keep both masked and active learning images in the training dataset; remove nothing
        pass

    else:
        # Remove both active learning images and masked images
        if ("al" in rapids_df.columns):
            rapids_df = rapids_df[(rapids_df["al"] == 0)]
        if ("masked" in rapids_df.columns):    
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

    classifier_config = ModelConfig(model_name=model_name,
                                    hidden_layers=(1024, 512),
                                    dropout=0.5,
                                    num_classes=2)

    # Initiate, train, and evaluate model
    classifier = RiverClassifier(classifier_config, artifact_dir, model_log_name)

    # We use an image size of 480 instead of the default image size of 224, 
    # since a larger image size appears to give a performance boost.
    image_size = 480 
    # The default image size can be used by uncommenting the line below
    # image_size = list(classifier.model.backbone.default_cfg.get("input_size"))[1]
    transform_mean = list(classifier.model.backbone.default_cfg.get("mean"))
    transform_std = list(classifier.model.backbone.default_cfg.get("std"))

    # Define transform now that we have the image size
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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
    train_dataset = RapidsDataset(image_dir, train_keys, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = RapidsDataset(image_dir, val_keys, val_labels, transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = RapidsDataset(image_dir, test_keys, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # ResNetv2
    # This needs to be commented out or modified if a different pretrained model architecture is used
    # Unfreeze only the final stage of the backbone and final norm
    for param in classifier.model.backbone.stages[3].parameters():
        param.requires_grad = True
    for param in classifier.model.backbone.norm.parameters():
        param.requires_grad = True

    # (Optional) Confirm classifier is trainable
    for param in classifier.model.classifier.parameters():
        param.requires_grad = True

    # Initialize the optimizer
    optim_config = OptimConfig(backbone_lr = 5e-5, classifier_lr = 1e-3, weight_decay = 1e-5, scheduler_patience = 5, scheduler_factor = 0.1)
    classifier.initialize_optimizer(optim_config)

    classifier.print_device()
    # Train the classifier model
    classifier.train(train_loader, val_loader, epochs=50, early_stop_patience=10)

    # Evaluate the model on the test data
    classifier.evaluate(test_loader)

    # Get predicted probabilities for the test data to calculate AUC
    # classifier.predict(test_loader)

if __name__ == '__main__':
    main()

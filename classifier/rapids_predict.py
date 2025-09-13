# Author name and contact: Nicholas Brimhall, ORCID: 0009-0008-7410-0166
#
# This file contains a script to predict the presence of rapids in a collection of satellite images of rivers. 
# The model uses the RiverClassifier class from the file classifier.py. Data is loaded using the WebDataset 
# library. Before running this script, the file paths below should be set to the actual locations of the files
# on the user's machine. The tar files containing river images should be downloaded from the CIRRUS data release
# and all placed in one directory (NOT extracted/unzipped). WebDataset is able to stream images directly from 
# uncompressed tar files. The image key and predicted probability of rapids are written to a CSV file
# in the directory specified by the artifact_dir variable below. 

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import webdataset as wds

# Set timm cache
os.environ['HF_HOME'] = os.path.join(Path.home(), "hf_cache")
from classifier import RiverClassifier

# Set the paths below to their respective locations
# Configuration --------------------------
# Path to model weights
model_path = os.path.join(Path.home(), "rapids", "model_performance", "best_resnetv2_152_model.pth")
# Path to directory containing one or more tar files with river images
image_dir = os.path.join(Path.home(), "rapids", "images")
# Directory where output CSV will be written
artifact_dir = os.path.join(Path.home(), "rapids", "model_predictions")
# File name of CSV to write predictions
model_log_name = "resnetv2_152_model"

model_type = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k"
image_size = 480
batch_size = 32

# Class to apply image normalization and resizing transforms
# Return None for label to ensure compatability with RiverClassifier predict method
class PreprocessSample:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = self.transform(sample["jpg"])
        label = None
        key = sample["__key__"]
        return image, label, key

# Dataclass for model architecture hyperparameters
@dataclass
class ModelConfig:
    model_name: str
    hidden_layers: Tuple[int, ...]
    dropout: float
    num_classes: int

def main():
    # Initialize the classifier
    classifier_config = ModelConfig(model_name=model_type,
                                    hidden_layers=(1024, 512),
                                    dropout=0.5,
                                    num_classes=2)
    classifier = RiverClassifier(classifier_config, artifact_dir, model_log_name)
    
    # Load the model weights
    classifier.model.load_state_dict(torch.load(model_path, map_location=classifier.device))

    classifier.model.eval()
    classifier.model.to(classifier.device)

    # Get image normalization values from the model  configuration
    # (provided for all timm models)
    transform_mean = list(classifier.model.backbone.default_cfg.get("mean"))
    transform_std = list(classifier.model.backbone.default_cfg.get("std"))
  
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
        ])
    
    # Set up WebDataset. This allows us to stream the images from all of the provided tar files
    # and is ideal since we only need to view each image once and we do not need to enforce
    # a train-test split for inference
    tar_files = ["".join(["file:", image_dir, "/", f]) for f in os.listdir(image_dir)]

    preprocess_sample = PreprocessSample(transform)

    dataset = (
        wds.WebDataset(tar_files, shardshuffle=False, empty_check=False)
        .decode("pil")
        .map(preprocess_sample)
        .batched(batch_size, partial=True)
    )

    # Wrap the WebDataset in a PyTorch DataLoader
    predict_loader = DataLoader(dataset, batch_size=None, num_workers=2)

    # Call predict method to predict on all images in predict_loader
    classifier.predict(predict_loader)
    
if __name__ == '__main__':
    main()

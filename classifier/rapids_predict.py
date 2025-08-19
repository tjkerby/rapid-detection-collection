import os
import csv
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import webdataset as wds

# Configuration --------------------------
MODEL_TYPE = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k"
IMAGE_SIZE = 480
MODEL_PATH = str(Path.home() / "rapids" / "best_resnetv2_07_08_full_model.pth")
IMAGE_DIR = str(Path.home() / "nbrim_images")
OUTPUT_CSV_PATH = str(Path.home() / "rapids" / "predictions_07_08.csv")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_LOG_NAME = ""

# Import timm and set cache
os.environ['HF_HOME'] = os.path.join(Path.home(), "hf_cache")
from classifier.py import RiverClassifier

# Class to apply image normalization and resizing transforms
class PreprocessSample:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = self.transform(sample["jpg"])
        key = sample["__key__"]
        return image, key

# Dataclass for model architecture hyperparameters
@dataclass
class ModelConfig:
    model_name: str
    hidden_layers: Tuple[int, ...]
    dropout: float
    num_classes: int

def main():
    # Initialize the classifier
    classifier_config = ModelConfig(model_name=MODEL_TYPE,
                                    hidden_layers=(1024, 512),
                                    dropout=0.5,
                                    num_classes=2)
    classifier = RiverClassifier(classifier_config, MODEL_LOG_NAME)
    
    # Load the model weights
    classifier.model.load_state_dict(torch.load(MODEL_PATH, map_location=classifier.DEVICE))

    classifier.model.eval()
    classifier.model.to(DEVICE)

    # Get image normalization values from the model  configuration
    # (provided for all timm models)
    transform_mean = list(classifier.model.backbone.default_cfg.get("mean"))
    transform_std = list(classifier.model.backbone.default_cfg.get("std"))
  
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
        ])
    
    # Set up WebDataset. This allows us to stream the images from all of the provided tar files
    # and is ideal since we only need to view each image once and we do not need to enforce
    # a train-test split for inference
    tar_files = ["".join(["file:", IMAGE_DIR, "/", f]) for f in os.listdir(IMAGE_DIR)]

    preprocess_sample = PreprocessSample(transform)

    dataset = (
        wds.WebDataset(tar_files, shardshuffle=False, empty_check=False)
        .decode("pil")
        .map(preprocess_sample)
        .batched(BATCH_SIZE, partial=True)
    )

    # Wrap the WebDataset in a PyTorch DataLoader
    predict_loader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Prepare a CSV to store the results and make predictions
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "predicted_prob"])
        print("Writing predictions to", OUTPUT_CSV_PATH)

        autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()
        with torch.no_grad(), autocast_ctx:
            for images, keys in predict_loader:
                images = images.to(DEVICE)

                outputs = classifier.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                for key, prob in zip(keys, probs):
                    writer.writerow([key, float(prob)])
                f.flush()

if __name__ == '__main__':
    main()

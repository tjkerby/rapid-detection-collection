import os
import csv
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import webdataset as wds

# Import timm and set cache
os.environ['HF_HOME'] = str(Path.home() / "hf_cache")
import timm

# Define RiverModel class
class RiverModel(nn.Module):
    def __init__(self, backbone, classifier_head, pooling=True):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier_head
        self.pooling = pooling

    def forward(self, x):
        # Use forward_features if it exists; else fallback to forward
        if hasattr(self.backbone, 'forward_features'):
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)

        # Global pooling if output is spatial (4D tensor)
        if self.pooling and x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Classifier head
        x = self.classifier(x)
        return x

# Define function to create classifier head
def make_classifier_head(in_features, hidden_layers=(1024, 512), dropout=0.5, num_classes=2, use_dropout_last=True):
    layers = []
    last_dim = in_features
    for i, h in enumerate(hidden_layers):
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU(inplace=True))

        last_dim = h

    if dropout > 0 and use_dropout_last:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(last_dim, num_classes))
    return nn.Sequential(*layers)

# Define function to get pretrained model with classifier head
def get_model(model_type, num_classes,
                hidden_layers=(1024, 512),
                dropout=0.5,
                freeze_backbone=True):
    model_name = model_type
    backbone = timm.create_model(model_name, pretrained=True)

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    # Determine input features for classifier head
    if hasattr(backbone, 'head') and hasattr(backbone.head, 'in_features'):
        in_features = backbone.head.in_features
    else:
        in_features = getattr(backbone, 'num_features', None)

    # Build shared classifier head
    classifier_head = make_classifier_head(
        in_features=in_features,
        hidden_layers=hidden_layers,
        dropout=dropout,
        num_classes=num_classes,
        use_dropout_last=True
    )

    return RiverModel(backbone, classifier_head)

class PreprocessSample:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = self.transform(sample["jpg"])
        key = sample["__key__"]
        return image, key

# Configuration --------------------------
MODEL_TYPE = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k" # "resnetv2_152x2_bit_teacher"
IMAGE_SIZE = 480
MODEL_PATH = str(Path.home() / "rapids" / "best_resnetv2_07_08_full_model.pth")
IMAGE_DIR = str(Path.home() / "nbrim_images")
OUTPUT_CSV_PATH = str(Path.home() / "rapids" / "predictions_07_08.csv")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Initialize model --------------------------
    model = get_model(MODEL_TYPE, 2)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.eval()
    model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Set up WebDataset --------------------------
    tar_files = ["".join(["file:", IMAGE_DIR, "/", f]) for f in os.listdir(IMAGE_DIR)]

    preprocess_sample = PreprocessSample(transform)

    dataset = (
        wds.WebDataset(tar_files, shardshuffle=False, empty_check=False)
        .decode("pil")
        .map(preprocess_sample)
        .batched(BATCH_SIZE, partial=True)
    )

    # Wrap in PyTorch DataLoader --------------------------
    predict_loader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Prepare output and make predictions --------------------------
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "predicted_prob"])
        print("Writing predictions to", OUTPUT_CSV_PATH)

        autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()
        with torch.no_grad(), autocast_ctx:
            for images, keys in predict_loader:
                images = images.to(DEVICE)

                outputs = model(images)
                if outputs.ndim > 2:
                    outputs = F.adaptive_avg_pool2d(outputs, 1).flatten(1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                for key, prob in zip(keys, probs):
                    writer.writerow([key, float(prob)])
                f.flush()

if __name__ == '__main__':
    main()

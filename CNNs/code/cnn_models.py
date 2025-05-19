import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import timm  # Add this import for new models

# Set HuggingFace cache directory before importing timm or any model
os.environ['HF_HOME'] = os.path.abspath('./hf_cache')

class StratifiedSampler:
    """
    Performs stratified sampling to maintain class distribution
    across train, validation, and test sets
    """
    @staticmethod
    def split_dataset(dataset, test_size=0.2, val_size=0.2, random_state=42):
        labels = np.array(dataset.targets)
        train_val_indices, test_indices, train_val_labels, _ = train_test_split(
            range(len(dataset)), labels, test_size=test_size, stratify=labels, random_state=random_state
        )
        train_indices, val_indices, _, _ = train_test_split(
            train_val_indices, train_val_labels,
            test_size=val_size / (1 - test_size), stratify=train_val_labels, random_state=random_state
        )
        return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


class RiverClassifier:
    def __init__(self, num_classes, model_type='resnet', learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._get_model(model_type, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=f'runs/{model_type}')
        self.model_type = model_type

    def _get_model(self, model_type, num_classes):
        if model_type == 'simple':
            return self._create_simple_cnn(num_classes)
        elif model_type == 'resnetpre':
            return self._create_resnetpre(num_classes)
        elif model_type == 'efficientnetpre':
            return self._create_efficientnetpre(num_classes)
        elif model_type == 'resnet':
            return self._create_resnet(num_classes)
        elif model_type == 'resnetv2':
            return self._create_resnetv2_large(num_classes)  # Use larger model
        elif model_type == 'efficientnet':
            return self._create_efficientnet(num_classes)
        elif model_type == 'efficientnetv2':
            return self._create_efficientnetv2_large(num_classes)  # New: EfficientNetV2
        elif model_type == 'vit':
            return self._create_vit_large(num_classes)  # Use larger model
        elif model_type == 'swin':
            return self._create_swin_large(num_classes)  # Use larger model
        elif model_type == 'convnext':
            return self._create_convnext_large(num_classes)  # Use larger model
        elif model_type == 'coatnet':
            return self._create_coatnet_large(num_classes)  # Use larger model
        else:
            raise ValueError("Invalid model type. Choose from 'simple', 'resnet', 'resnetv2', 'efficientnet', 'efficientnetv2', 'vit', 'swin', 'convnext', or 'coatnet'")

    def _create_simple_cnn(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 640x640
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 320x320
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 160x160
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 80x80
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 40x40
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x20
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10
            nn.AdaptiveAvgPool2d(1),  # 1x1
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _create_resnetpre(self, num_classes):
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return resnet

    def _create_resnet(self, num_classes):
        resnet = models.resnet152(pretrained=False)
        for param in resnet.parameters():
            param.requires_grad = False
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return resnet

    def _create_efficientnetpre(self, num_classes):
        efficientnet = models.efficientnet_b7(pretrained=True)
        for param in efficientnet.parameters():
            param.requires_grad = False
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return efficientnet

    def _create_efficientnet(self, num_classes):
        efficientnet = models.efficientnet_b7(pretrained=False)
        for param in efficientnet.parameters():
            param.requires_grad = False
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return efficientnet

    def _create_resnetv2(self, num_classes):
        # ResNet152v2 using timm
        model = timm.create_model('resnetv2_152x2_bit_teacher', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_vit(self, num_classes):
        # Vision Transformer (ViT) using timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Sequential(
            nn.Linear(model.head.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_swin(self, num_classes):
        # Swin Transformer using timm
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Sequential(
            nn.Linear(model.head.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_convnext(self, num_classes):
        # ConvNeXt using timm
        model = timm.create_model('convnext_tiny', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_coatnet(self, num_classes):
        # CoAtNet using timm
        model = timm.create_model('coatnet_0_rw_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    # Add new methods for larger models
    def _create_resnetv2_large(self, num_classes):
        # Use a valid large ResNetV2 model from timm
        model = timm.create_model('resnetv2_152x2_bit_teacher', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Remove the original head and add a new one that expects pooled features
        if hasattr(model, 'head'):
            # Remove the head entirely and replace with identity
            model.head = nn.Identity()
        # Add a new classifier after global pooling
        model.classifier = nn.Sequential(
            nn.Linear(model.num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        # Wrap the model to apply global pooling and classifier after the backbone
        class ResNetV2Custom(nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier
            def forward(self, x):
                x = self.backbone.forward_features(x)
                # Global average pool if needed
                if x.ndim == 4:
                    x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                return self.classifier(x)
        return ResNetV2Custom(model, model.classifier)

    def _create_vit_large(self, num_classes):
        # Use a larger ViT model
        model = timm.create_model('vit_large_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Sequential(
            nn.Linear(model.head.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_swin_large(self, num_classes):
        # Use a larger Swin Transformer model
        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Sequential(
            nn.Linear(model.head.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_convnext_large(self, num_classes):
        # Use a larger ConvNeXt model
        model = timm.create_model('convnext_large_in22ft1k', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_coatnet_large(self, num_classes):
        # Use a larger CoAtNet model
        model = timm.create_model('coatnet_3_rw_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def _create_efficientnetv2_large(self, num_classes):
        # EfficientNetV2-L, use pretrained weights if available, else fallback to random init
        try:
            model = timm.create_model('efficientnetv2_l', pretrained=True)
        except RuntimeError:
            model = timm.create_model('efficientnetv2_l', pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return model

    def train(self, train_loader, val_loader, epochs=50, early_stop_patience=15):
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses, val_losses, val_accuracies = [], [], []
        best_model_path = f'best_{self.model_type}_model.pth'
        completed_epochs = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(images)
                    # If output is not [B, C], apply global pooling and flatten
                    if outputs.ndim > 2:
                        outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()

            val_loss, val_accuracy = self._evaluate(val_loader)
            self.scheduler.step(val_loss)

            # Store metrics
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            self.writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    completed_epochs = epoch + 1
                    break
            completed_epochs = epoch + 1

        self.model.load_state_dict(torch.load(best_model_path))
        self._save_training_plot(train_losses[:completed_epochs], 
                               val_losses[:completed_epochs], 
                               val_accuracies[:completed_epochs], 
                               completed_epochs)

    def _save_training_plot(self, train_losses, val_losses, val_accuracies, epochs):
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(f'{self.model_type}_training_plot_train.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig(f'{self.model_type}_training_plot_validation.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(f'{self.model_type}_training_plot_accuracy.png')
        plt.close()

    def _evaluate(self, loader):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # If output is not [B, C], apply global pooling and flatten
                if outputs.ndim > 2:
                    outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)
                val_loss += self.criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return val_loss / len(loader), 100 * correct / total

    def evaluate(self, test_loader, class_names):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # If output is not [B, C], apply global pooling and flatten
                if outputs.ndim > 2:
                    outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        # Ensure all classes are represented in the confusion matrix and report
        labels_unique = np.unique(np.concatenate([all_labels, all_preds]))
        # Only use valid indices for class_names
        valid_labels = [i for i in labels_unique if 0 <= i < len(class_names)]
        cm = confusion_matrix(all_labels, all_preds, labels=valid_labels)
        # Avoid division by zero in confusion matrix normalization
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm, row_sums, where=row_sums != 0)
        display_names = [class_names[i] for i in valid_labels]
        sns.heatmap(norm_cm, annot=True, fmt='.2%', xticklabels=display_names, yticklabels=display_names)
        plt.savefig(f'{self.model_type}_confusion_matrix.png')
        plt.close()
        print(classification_report(all_labels, all_preds, labels=valid_labels, target_names=display_names))
        return classification_report(all_labels, all_preds, labels=valid_labels, target_names=display_names, output_dict=True)

# Restore correct image sizes for each model (these are the official input sizes for each architecture)
MODEL_IMAGE_SIZES = {
    'simple': 1280,         # Custom model, keep large
    'resnet': 224,          # torchvision/models/resnet
    'resnetpre': 224,       # torchvision/models/resnet
    'efficientnet': 600,    # torchvision/models/efficientnet_b7
    'efficientnetpre': 600, # torchvision/models/efficientnet_b7
    'efficientnetv2': 480,  # timm efficientnetv2_l
    'resnetv2': 480,        # timm resnetv2_152x4_bit_teacher
    'vit': 224,             # timm vit_large_patch16_224
    'swin': 224,            # timm swin_large_patch4_window7_224
    'convnext': 224,        # timm convnext_large_in22ft1k
    'coatnet': 224,         # timm coatnet_3_rw_224
}

def prepare_datasets(data_dir, image_size=1280, batch_size=4, random_state=42):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=data_dir, transform=train_transform)
    train_subset, val_subset, test_subset = StratifiedSampler.split_dataset(dataset, test_size=0.2, val_size=0.2, random_state=random_state)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, dataset.classes

def main():
    data_dir = './dataset'  # Replace with your data directory
    for model in [ 'efficientnetpre','efficientnetv2', 'resnetv2', 'vit', 'swin', 'convnext', 'coatnet']:
        print(f"Training {model} model")
        print("=====================================")
        image_size = MODEL_IMAGE_SIZES.get(model, 224)
        train_loader, val_loader, test_loader, class_names = prepare_datasets(data_dir, image_size=image_size)
        classifier = RiverClassifier(num_classes=len(class_names), model_type=model)
        classifier.train(train_loader, val_loader, epochs=100)
        classifier.evaluate(test_loader, class_names)

if __name__ == '__main__':
    main()

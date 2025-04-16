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


class TrashClassifier:
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
        elif model_type == 'efficientnet':
            return self._create_efficientnet(num_classes)
        else:
            raise ValueError("Invalid model type. Choose 'simple', 'resnet', or 'efficientnet'")

    def _create_simple_cnn(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.AdaptiveAvgPool2d(1),  
            
            nn.Flatten(),  
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def _create_resnetpre(self, num_classes):
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return resnet
    def _create_resnet(self, num_classes):
        resnet = models.resnet18(pretrained=False)
        for param in resnet.parameters():
            param.requires_grad = False
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return resnet

    def _create_efficientnetpre(self, num_classes):
        efficientnet = models.efficientnet_b0(pretrained=True)
        for param in efficientnet.parameters():
            param.requires_grad = False
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return efficientnet
    def _create_efficientnet(self, num_classes):
        efficientnet = models.efficientnet_b0(pretrained=False)
        for param in efficientnet.parameters():
            param.requires_grad = False
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        return efficientnet

    def train(self, train_loader, val_loader, epochs=50, early_stop_patience=7):
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(images)
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
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    # break

        # Save the training plot after training or early stopping
        # torch.save(self.model.state_dict(), f'finished_{self.model_type}_model.pth')
        self._save_training_plot(train_losses, val_losses, val_accuracies, epochs)


    def _save_training_plot(self, train_losses, val_losses, val_accuracies, epochs):
        epochs_range = range(1, epochs +1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(f'{self.model_type}_training_plot_train.png')
        plt.close()
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig(f'{self.model_type}_training_plot_validation.png')
        plt.close()
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
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%', xticklabels=class_names, yticklabels=class_names)
        plt.savefig(f'{self.model_type}_confusion_matrix.png')
        plt.close()
        print(classification_report(all_labels, all_preds, target_names=class_names))

        return classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)


def prepare_datasets(data_dir, image_size=512, batch_size=32, random_state=42):
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

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, dataset.classes


def main():
    data_dir = './data'  # Replace with your data directory
    train_loader, val_loader, test_loader, class_names = prepare_datasets(data_dir)
    for model in [ 'simple','resnet', 'efficientnet', 'resnetpre', 'efficientnetpre']:
        print(f"Training {model} model")
        print("=====================================")
        classifier = TrashClassifier(num_classes=len(class_names), model_type=model)
        classifier.train(train_loader, val_loader, epochs=50)
        classifier.evaluate(test_loader, class_names)
    # classifier = TrashClassifier(num_classes=len(class_names), model_type='efficientnet')
    # classifier.train(train_loader, val_loader, epochs=20)
    # classifier.evaluate(test_loader, class_names)


if __name__ == '__main__':
    main()

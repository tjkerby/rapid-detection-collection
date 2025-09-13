# Author name and contact: Nicholas Brimhall, ORCID: 0009-0008-7410-0166
#
# This file contains a class definition for a RiverClassifier deep learning model, which can be used to predict
# the presence of rapids in satellite imagery of rivers. The model itself is implemented using PyTorch. A pretrained
# convolutional neural network (CNN) model obtained from the timm library is used as the primary 
# backbone for this model. This class was designed and tested to work with a ResNetv2_152 model, but
# there is some flexibility in implementing other pretrained networks as well. Mixed precision is implemented
# to speed up training on machines where CUDA is available. 

import csv
import os 
from contextlib import nullcontext
from dataclasses import dataclass

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import matplotlib.pyplot as plt

import timm

# Class containing the classifier model, optimizer, and training and evaluation methods
class RiverClassifier:
    # model_config: A ModelConfig dataclass containing the hyperparameters for the classifier
    # artifacts_dir: Path to a directory to store model outputs
    # model_log_name: The file name used for saving results and model weights
    def __init__(self, model_config, artifacts_dir, model_log_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config
        self.artifacts_dir = artifacts_dir
        self.model_log_name = model_log_name
        self.model = RiverClassifier.get_model(model_name=model_config.model_name,
                                               hidden_layers=model_config.hidden_layers,
                                               dropout=model_config.dropout,
                                               num_classes=model_config.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()

    def print_device(self):
        print("Using device: %s" % self.device)

    # Model wrapper class used to connect pretrained CNN with classifier head
    # Initialized within get_model--should not be initialized directly
    class RiverModel(nn.Module):
        # backbone: Pretrained CNN model from timm
        # classifier_head: A feedforward classifier from make_classifier_head
        # pooling: Should average pooling be applied if necessary
        def __init__(self, backbone, classifier_head, pooling=True):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier_head
            self.pooling = pooling

        def forward(self, x):
            # Use forward_features if it exists; else fallback to forward
            if hasattr(self.backbone, "forward_features"):
                x = self.backbone.forward_features(x)
            else:
                x = self.backbone(x)

            # Global pooling if output is spatial (4D tensor)
            if self.pooling and x.ndim == 4:
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)

            # Classifier head
            x = self.classifier(x)
            return x

    # Method for creating feedforward classifier head. Called by get_model
    # in_features: Number of input features (i.e. number of output features from
    # the final layer of the convolutional network, after pooling). Automatically
    # determined within get_model
    # dropout: Value for a dropout layer after the final hidden layer. If 0, then
    # a dropout layer is not used.
    # num_classes: Number of classes, i.e. output neurons.
    # use_dropout_last: Should a dropout layer be used after the final hidden layer
    @staticmethod
    def make_classifier_head(in_features, hidden_layers, dropout, num_classes):
        layers = []
        last_dim = in_features
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))

            last_dim = h

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(last_dim, num_classes))
        return nn.Sequential(*layers)

    # Method to create a classifier model by attaching a custom classifier head
    # to a pretrained convolutional network
    # model_name: Model name, in timm, of a pretrained CNN
    # hidden_layers: Number of hidden layers in the classifier head
    # dropout: Value between 0 and 1 for dropout layers
    # num_classes: Number of classes, i.e. output neurons.
    # freeze_backbone: Should the pretrained parameters have requires_grad disabled by default
    # (i.e. unless specific blocks are unfrozen later to fine-tune the final stages of the model)
    @staticmethod
    def get_model(model_name, hidden_layers, dropout, num_classes, freeze_backbone=True):
        backbone = timm.create_model(model_name, pretrained=True)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Determine input features for classifier head
        # Various timm models have slightly different naming conventions for
        # the model features
        if hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
            in_features = backbone.head.in_features
            # backbone.head = nn.Identity()
        if hasattr(backbone, "fc") and hasattr(backbone.fc, "in_features"):
            in_features = backbone.head.in_features
            # backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier") and hasattr(backbone.classifier, "in_features"):
            in_features = backbone.classifier.in_features
            # backbone.classifier = nn.Identity()
        else:
            in_features = getattr(backbone, "num_features", None)

        # Build shared classifier head
        classifier_head = RiverClassifier.make_classifier_head(
            in_features=in_features,
            hidden_layers=hidden_layers,
            dropout=dropout,
            num_classes=num_classes
        )

        return RiverClassifier.RiverModel(backbone, classifier_head)

    # Method for initializing an AdamW optimizer using an OptimConfig dataclass
    def initialize_optimizer(self, optim_config):
        backbone_params = filter(lambda p: p.requires_grad, self.model.backbone.parameters())
        classifier_params = filter(lambda p: p.requires_grad, self.model.classifier.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": optim_config.backbone_lr},
            {"params": classifier_params, "lr": optim_config.classifier_lr},
        ], weight_decay=optim_config.weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=optim_config.scheduler_patience, factor=optim_config.scheduler_factor)

    # Dataclass used to report values for a confusion matrix
    # Used by the protected _evaluate method
    @dataclass
    class EvalMetrics:
        TP: int
        FP: int
        TN: int
        FN: int
        precision: float
        recall: float
        f1: float
        accuracy: float
        eval_loss: float

    # Method for training the classifier model. This method implements early stopping
    # using the validation F1 metric, saving the current model state after a new
    # best validation F1 is recorded and halting training after early_stop_patience
    # epochs
    # train_loader: A DataLoader of training images
    # val_loader: A DataLoader of validation images
    # epochs: Maximum number of epochs over which to train the model
    # early_stop_patience: Number of epochs for early stopping
    # save_model: Should the model be saved at the best validation loss (True)
    # or save at the final epoch (False)
    def train(self, train_loader, val_loader, epochs, early_stop_patience, save_best_model=True):
        # Initialize best validation F1 score to 0
        # If validation loss is being used for early stopping, this should be set
        # to the maximum float value.
        best_val_metric = 0
        early_stop_counter = 0
        # Lists of training and validation metrics per epoch, used for plots
        train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []
        best_model_path = os.path.join(self.artifacts_dir, "".join(["best_", self.model_log_name, "_model.pth"]))
        completed_epochs = 0

        print("#--------------------------------------------------------------")
        print("Training Model: %s" %(self.model_config.model_name))

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                autocast_ctx = autocast(device_type="cuda") if self.device == "cuda" else nullcontext()
                with autocast_ctx:
                    outputs = self.model(images)

                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()

            val_metrics = self._evaluate(val_loader)
            self.scheduler.step(val_metrics.eval_loss)

            # Store metrics
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_metrics.eval_loss)
            val_accuracies.append(val_metrics.accuracy)
            val_f1s.append(val_metrics.f1)

            print("Epoch %d of %d, Train Loss: %.4f, Validation Loss: %.4f, Validation Accuracy: %.2f, Validation F1: %.2f"
                   %(epoch + 1, epochs, train_loss, val_metrics.eval_loss, val_metrics.accuracy, val_metrics.f1))

            completed_epochs = epoch + 1

            # Logic for early stopping using the validation F1 score
            if val_metrics.f1 > best_val_metric:
                best_val_metric = val_metrics.f1
                early_stop_counter = 0
                best_model_state = self.model.state_dict()
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    break

        # Load the best model state for evaluation and save to a .pth file
        if (save_best_model):
            self.model.load_state_dict(best_model_state)
            torch.save(best_model_state, best_model_path)
        else:
            torch.save(self.model.state_dict(), best_model_path)
        self._save_training_plot(train_losses, val_losses, val_accuracies, val_f1s, completed_epochs)

    # Method to save line plots for training and validation metrics vs epoch
    def _save_training_plot(self, train_losses, val_losses, val_accuracies, val_f1s, epochs):
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, train_losses[:epochs], label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(self.artifacts_dir, "".join([self.model_log_name, "_training_plot_train.png"])))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, val_losses[:epochs], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.artifacts_dir, "".join([self.model_log_name, "_training_plot_validation.png"])))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, val_accuracies[:epochs], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.artifacts_dir, "".join([self.model_log_name, "_training_plot_accuracy.png"])))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, val_f1s[:epochs], label='Validation F1')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('F1')
        plt.legend()
        plt.savefig(os.path.join(self.artifacts_dir, "".join([self.model_log_name, "_training_plot_f1.png"])))
        plt.close()

    # Method for evaluating the performance of the model on the test or validation data. 
    # loader: A DataLoader of river images returning image, label, and key
    # Returns: an EvalMetrics dataclass with the information necessary to generate a
    # confusion matrix 
    def _evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                autocast_ctx = autocast(device_type="cuda") if self.device == "cuda" else nullcontext()
                with autocast_ctx:
                    outputs = self.model(images)

                    loss = self.criterion(outputs, labels)

                eval_loss += loss.item() # self.criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds.cpu())
                all_labels.extend(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.tensor(all_labels)

        TP = ((all_preds == 1) & (all_labels == 1)).sum().item()
        TN = ((all_preds == 0) & (all_labels == 0)).sum().item()
        FP = ((all_preds == 1) & (all_labels == 0)).sum().item()
        FN = ((all_preds == 0) & (all_labels == 1)).sum().item()

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy  = (TP + TN) / (TP + TN + FP + FN)

        return RiverClassifier.EvalMetrics(TP, FP, TN, FN, precision, recall, f1, accuracy, eval_loss / len(loader))

    # Method to provide an exposed wrapper around the protected _evaluate method and save the metrics to a CSV
    # loader: A DataLoader of river images returning image, label, and key
    # save: Should the returned metrics be written to a CSV. Use save=False to avoid overwriting existing results
    def evaluate(self, loader, save=True):
        em = self._evaluate(loader)

        if (save):
            results_path = os.path.join(self.artifacts_dir, self.model_log_name) + ".csv"
            f = open(results_path, "w", newline="")
            f.write("TP,FP,TN,FN,precision,recall,f1,accuracy\n")
            f.write("%d,%d,%d,%d,%f,%f,%f,%f\n" %(em.TP, em.FP, em.TN, em.FN, em.precision, em.recall, em.f1, em.accuracy))
            f.close()

        print("Test Loss: %.4f, Test Accuracy: %.4f, Test F1: %.4f, Test Precision: %.4f, Test Recall: %.4f" 
              %(em.eval_loss, em.accuracy, em.f1, em.precision, em.recall))

    # Method to predict the probability of the presence of a rapid in a collection of river images.  
    # Writes a CSV containing the image key and the predicted probability. This method is used in
    # rapids_predict.py to perform inference on a large collection of unlabeled river images. 
    # loader: A DataLoader of river images returning image, label, and key. Label is required to
    # maintain compatibility with the Dataset class used to train the model, but is not used in 
    # making or writing the predictions. The DataLoader used in rapids_predict.py is configured
    # to return None for each label, which is then discarded.
    def predict(self, loader):
        csv_path = os.path.join(self.artifacts_dir, self.model_log_name) + "_preds.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "predicted_prob"])
            print("Writing predictions to", csv_path)

            autocast_ctx = torch.amp.autocast(device_type="cuda") if self.device == "cuda" else nullcontext()
            with torch.no_grad(), autocast_ctx:
                for images, _, keys in loader:
                    images = images.to(self.device)

                    outputs = self.model(images)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                    for key, prob in zip(keys, probs):
                        writer.writerow([key, float(prob)])
                    f.flush()

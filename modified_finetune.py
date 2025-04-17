import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import shutil

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device

def setup_directories():
    rootPath = Path('C:/Users/kaden/Box/STAT5810/rapids/data')
    CACHE_DIR = Path('cache')
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / 'images').mkdir(exist_ok=True)
    (CACHE_DIR / 'masks').mkdir(exist_ok=True)
    return rootPath, CACHE_DIR

def load_data(rootPath):
    image_dir = rootPath / 'images'
    mask_dir = rootPath / 'masks'
    
    # Get all mask files and verify corresponding images exist
    mask_files = [f.stem for f in mask_dir.glob('*.npy')]
    data = [f for f in mask_files if (image_dir / f"{f}.png").exists()]
    
    print(f"Total masks found: {len(mask_files)}")
    print(f"Total valid pairs (mask + image): {len(data)}")
    
    # Split the data
    indices = range(len(data))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    return train_data, test_data

def setup_model(device):
    sam2_checkpoint = "maskingTool/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    return sam2_model, predictor

def read_batch(data, index, rootPath, CACHE_DIR):
    # Sanitize filename by replacing problematic characters
    safe_filename = data[index].replace(':', '_').replace(' ', '_')
    cache_image_path = CACHE_DIR / "images" / f"{safe_filename}.png"
    cache_mask_path = CACHE_DIR / "masks" / f"{safe_filename}.npy"
    
    # Check cache first
    if cache_image_path.exists() and cache_mask_path.exists():
        try:
            image = Image.open(str(cache_image_path))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            mask = np.load(str(cache_mask_path))
            return image, mask
        except Exception as e:
            print(f"Error loading from cache: {e}")
            # If cache is corrupted, fall through to original source
    
    # If not in cache, load from Box
    try:
        image_path = rootPath / "images" / f"{safe_filename}.png"
        mask_path = rootPath / "masks" / f"{safe_filename}.npy"
        
        image = Image.open(str(image_path))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        mask = np.load(str(mask_path))
        
        # Save to cache
        image.save(str(cache_image_path))
        np.save(str(cache_mask_path), mask)
        
        return image, mask
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

# Add error handling for batch loading
def safe_read_batch(data, index):
    img, mask = read_batch(data, index)
    if img is None or mask is None:
        # Skip problematic files
        return None, None
    return img, mask

# Import and load the SAM2 model (using the tiny version)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class RiverDataset(Dataset):
    def __init__(self, data_list, rootPath, CACHE_DIR):
        self.data_list = data_list
        self.rootPath = rootPath
        self.CACHE_DIR = CACHE_DIR
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        img, mask = read_batch(self.data_list, idx, self.rootPath, self.CACHE_DIR)
        if img is None or mask is None:
            return None
        return img, mask

def collate_fn(batch):
    # Filter out None values
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # Convert zip to lists
    imgs, masks = list(zip(*batch))
    return list(imgs), list(masks)

def train_model(sam2_model, predictor, train_loader, test_data, device, rootPath, CACHE_DIR, num_epochs=10):
    optimizer = optim.Adam(params=sam2_model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Allow the model to predict with no input points
    input_point = np.empty((0, 2)) 
    input_label = np.empty((0,), dtype=int)
    
    for epoch in range(num_epochs):
        sam2_model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            if batch is None:  # Skip bad batches
                continue
            imgs, masks = batch
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for img, mask in zip(imgs, masks):
                predictor.set_image(img)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                   points=(unnorm_coords, labels), boxes=None, masks=None,
                )
                
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                # Calculate the loss for the segmentation
                gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                # Calculate the loss for the score
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Combine the two losses, segmentation loss has a much higher weight
                loss = seg_loss + score_loss * 0.05

                batch_loss += loss
            
            batch_loss = batch_loss / len(imgs)  # Average loss over batch
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(sam2_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Validation phase
        sam2_model.eval()
        val_loss = 0
        with torch.no_grad():
            for j in range(len(test_data)):
                img, mask = read_batch(test_data, j, rootPath, CACHE_DIR)
                predictor.set_image(img)
                pred_masks, scores, logits = predictor.predict(
                    point_coords = input_point,
                    point_labels = input_label,
                    multimask_output = False
                )

                # Calculate the loss for the segmentation
                gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                # Calculate the loss for the score
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Combine the two losses, segmentation loss has a much higher weight
                loss = seg_loss + score_loss * 0.05
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_data)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save only one model file with best validation loss
            torch.save(sam2_model.state_dict(), f"sam2_model_finetuned_april_16.pt")
            patience_counter = 0
            print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
            
        print(f"Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # No need to load/save at end since we're already keeping the best model
    print(f"Training complete! Best validation loss: {best_loss:.4f}")

def main():
    device = setup_device()
    rootPath, CACHE_DIR = setup_directories()
    train_data, test_data = load_data(rootPath)
    sam2_model, predictor = setup_model(device)
    
    train_dataset = RiverDataset(train_data, rootPath, CACHE_DIR)
    batch_size = 32
    num_workers = 8
    
    # Print batch information
    num_batches = len(train_data) // batch_size + (1 if len(train_data) % batch_size != 0 else 0)
    print(f"\nDataset info:")
    print(f"Total training samples: {len(train_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {num_batches}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)
    
    # Set higher number of epochs for small dataset, early stopping will prevent overfitting
    train_model(sam2_model, predictor, train_loader, test_data, device, rootPath, CACHE_DIR, num_epochs=75)

if __name__ == '__main__':
    mp.freeze_support()
    main()
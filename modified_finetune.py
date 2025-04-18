import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
import json
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader

rootPath = './cache/'
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
# Function to read a batch from a json and return the image and mask in a format usable by the SAM2 model
def read_batch(data, index):
    image = Image.open(rootPath + "images/" + data[index] + ".png")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    mask = np.load(rootPath +  "masks/" + data[index] + ".npy")
    return image, mask

mask_dir = rootPath + 'masks'
data = [f.replace('.npy', '') for f in os.listdir(mask_dir) if f.endswith(('.npy'))]

# Split the data into training and testing sets
indices = range(len(data))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_data = [data[i] for i in train_indices]
test_data = [data[i] for i in test_indices]

# filter out invalid samples up front
train_data = [
    d for d in train_data
    if not (
        np.all(np.load(rootPath + "masks/" + d + ".npy") == 0)
        or np.all(np.load(rootPath + "masks/" + d + ".npy") == 1)
    )
]
test_data = [
    d for d in test_data
    if not (
        np.all(np.load(rootPath + "masks/" + d + ".npy") == 0)
        or np.all(np.load(rootPath + "masks/" + d + ".npy") == 1)
    )
]

class MaskDataset(Dataset):  # Change from data.Dataset to Dataset
    def __init__(self, ids): 
        self.ids = ids
    def __len__(self): 
        return len(self.ids)
    def __getitem__(self, idx): 
        return read_batch(self.ids, idx)

def collate_fn(batch):
    return batch  # keep (PIL, np.ndarray) pairs in a list

batch_size = 16
train_loader = DataLoader(  # Change from data.DataLoader to DataLoader
    MaskDataset(train_data),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(  # Change from data.DataLoader to DataLoader
    MaskDataset(test_data),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Import and load the SAM2 model (using the tiny version)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "maskingTool/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

optimizer = optim.Adam(params=sam2_model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
# epsilon = 0                        # clamp/IOU epsilon

epsilon = 1e-6                         # clamp/IOU epsilon
# Allow the model to predict with no input points
input_point = np.empty((0, 2)) 
input_label = np.empty((0,), dtype=int)

num_epochs = 50
for epoch in range(num_epochs):
    sam2_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        batch_loss = 0.0
        for img, mask in batch:
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
            prd_mask = prd_mask.clamp(epsilon, 1 - epsilon)  # prevent log(0)/1
            seg_loss = (-gt_mask * torch.log(prd_mask) 
                        - (1 - gt_mask) * torch.log(1 - prd_mask)).mean()

            # Calculate the loss for the score
            inter = (gt_mask * (prd_mask > 0.5)).sum((1,2))
            denom = gt_mask.sum((1,2)) + (prd_mask > 0.5).sum((1,2)) - inter
            iou = (inter + epsilon) / (denom + epsilon)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

            # Combine the two losses, segmentation loss has a much higher weight
            loss = seg_loss + score_loss * 0.05

            batch_loss += loss
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()

    # evaluate on test set after each epoch
    sam2_model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            for img, mask in batch:
                predictor.set_image(img)
                pred_masks, scores, _ = predictor.predict(
                    point_coords=input_point, point_labels=input_label, multimask_output=False
                )

                # Calculate the loss for the segmentation
                gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                prd_mask = torch.tensor(pred_masks[0], dtype=torch.float32, device=device)
                prd_mask = prd_mask.clamp(epsilon, 1 - epsilon)
                seg_loss = (-gt_mask * torch.log(prd_mask) 
                            - (1 - gt_mask) * torch.log(1 - prd_mask)).mean()
                if seg_loss.isnan():
                    print(f"Seg loss is NaN")

                # Calculate the loss for the score
                inter = (gt_mask * (prd_mask > 0.5)).sum((0,1))
                denom = gt_mask.sum() + (prd_mask > 0.5).sum() - inter
                iou = (inter + epsilon) / (denom + epsilon)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                if score_loss.isnan():
                    print(f"Score loss is NaN")
                # Combine the two losses, segmentation loss has a much higher weight
                loss = seg_loss + score_loss * 0.05
                test_losses.append(loss.item())
    print(
        f"Epoch {epoch}: Train Loss {batch_loss.item():.4f}, "
        f"Test Loss {np.mean(test_losses):.4f}"
    )
    # Save after each epoch
    torch.save(sam2_model.state_dict(), f"sam2_model_finetuned_epoch_april_18_2.pt")
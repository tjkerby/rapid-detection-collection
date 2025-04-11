import os
import concurrent.futures
import pandas as pd
from PIL import Image
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

boxPath = '/Users/cameronswapp/Library/CloudStorage/Box-Box/STAT5810/rapids/data/'

imgPath = boxPath + 'images/'
modelPath = '/Users/cameronswapp/Documents/STAT5810_ML/rapid-detection-collection/sam2_model_finetuned_2.pt'
modelPathOld = '/Users/cameronswapp/Documents/STAT5810_ML/rapid-detection-collection/sam2_model_finetuned_epoch_3.pt'

newMaskPath = "/Users/cameronswapp/Documents/STAT5810_ML/rapid-detection-collection/masks_computer"
    
imgFiles = [os.path.join(imgPath, f) for f in os.listdir(imgPath) if f.endswith('.png')][:50]

def compute_metadata(imgPath):
    pred_masks, scores, logits = predictMask(imgPath)
    # pred_masks_old, scores_old, logits_old = predictMask_old(imgPath)
    maskName = os.path.basename(imgPath).replace('.png', '.npy')
    maskPath = os.path.join(newMaskPath, maskName)
    np.save(maskPath, pred_masks)

    return {
        'img': imgPath,
        'score_new': scores,
        'mask_new': maskPath #,
        # 'score_old': scores_old,
        # 'mask_old': pred_masks_old,
    }

def predictMask(img, input_point=np.empty((0, 2)), input_label=np.empty((0,), dtype=int)):
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    # Load the model state that we trained
    predictor.model.load_state_dict(torch.load(modelPath))

    # Ensure each image is in RGB mode
    img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Predict the mask for each image
    predictor.set_image(img)
    pred_masks, scores, logits = predictor.predict(
        point_coords = input_point,
        point_labels = input_label,
        multimask_output = False
     )
    return pred_masks, scores, logits

def predictMask_old(img, input_point=np.empty((0, 2)), input_label=np.empty((0,), dtype=int)):
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    # Load the model state that we trained
    predictor.model.load_state_dict(torch.load(modelPathOld))

    # Ensure each image is in RGB mode
    img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Predict the mask for each image
    predictor.set_image(img)
    pred_masks, scores, logits = predictor.predict(
        point_coords = input_point,
        point_labels = input_label,
        multimask_output = False
     )
    return pred_masks, scores, logits

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_metadata, imgFiles))

    df = pd.DataFrame(results)
    df.to_csv('segmentation_results.csv', index=False)

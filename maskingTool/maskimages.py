import torch
import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import argparse
from omegaconf import DictConfig, OmegaConf

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Process images in a directory to create masks.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output masks')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Directory containing SAM2 model checkpoints')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for segmentation')
    return parser.parse_args()

def main(cfg: DictConfig):
    # Access configuration parameters from cfg instead of command line arguments
    input_dir = cfg.get('input_dir', 'input')
    output_dir = cfg.get('output_dir', 'output')
    checkpoint_dir = cfg.get('checkpoint_dir', '../checkpoints')
    threshold = cfg.get('threshold', 0.5)
    
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    print(f"Processing images from: {input_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Ensure output directory exists
    create_directory(output_dir)
    
    # Load model
    print("Loading SAM2 model...")
    model = load_model(checkpoint_dir)
    
    # Process images
    print("Processing images...")
    process_images(input_dir, output_dir, model, threshold=threshold)
    
    print("Done!")

def load_model(checkpoint_dir):
    # Use CUDA directly instead of select_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Using path format from segmentation.py
    model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'
    sam2_model = build_sam2(model_cfg, f'{checkpoint_dir}/sam2.1_hiera_tiny.pt', device=device)
    model = SAM2ImagePredictor(sam2_model)
    model.model.load_state_dict(torch.load(f'{checkpoint_dir}/sam2_model_finetuned_2.pt'))
    return model

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def process_images(input_dir, output_dir, model, threshold=0.5):
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob(os.path.join(input_dir, ext)))
        image_files.extend(glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Extract filename without extension
        filename = os.path.basename(image_path)
        base_filename, _ = os.path.splitext(filename)
        
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in model
        model.set_image(image)
        
        # Generate mask using automatic segmentation (no points)
        masks, scores, _ = model.predict()
        
        # If any mask found above threshold
        if len(masks) > 0 and max(scores) > threshold:
            # Get best mask
            best_idx = scores.argmax()
            mask = masks[best_idx].astype(bool)  # Ensure mask is boolean type
            
            # Save mask as binary image
            mask_output_path = os.path.join(output_dir, f"{base_filename}_mask.png")
            cv2.imwrite(mask_output_path, (mask * 255).astype(np.uint8))
            
            # Optional: Save visualization
            masked_img = image.copy()
            # Apply color overlay where mask is True
            masked_img[mask] = masked_img[mask] * 0.7 + np.array([255, 0, 0]) * 0.3
            vis_output_path = os.path.join(output_dir, f"{base_filename}_visualization.png")
            cv2.imwrite(vis_output_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Create a simple config dictionary to pass to main
    cfg = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'threshold': args.threshold
    }
    
    # Convert to DictConfig for compatibility
    cfg_dict = OmegaConf.create(cfg)
    
    # Call main with the config
    main(cfg_dict)

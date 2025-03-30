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
    model_cfg = '/configs/sam2.1/sam2.1_hiera_t.yaml'
    sam2_model = build_sam2(model_cfg, f'{checkpoint_dir}/sam2.1_hiera_tiny.pt', device=device)
    model = SAM2ImagePredictor(sam2_model)
    model.model.load_state_dict(torch.load(f'{checkpoint_dir}/sam2_model_finetuned_2.pt'))
    # model.model.load_state_dict(torch.load(f'{checkpoint_dir}/sam2_model_finetuned_epoch_3.pt'))

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
    
    # Create subdirectories for masks and masked images
    masks_dir = os.path.join(output_dir, "masks")
    masked_images_dir = os.path.join(output_dir, "masked_images")
    low_conf_dir = os.path.join(output_dir, "low_confidence")
    create_directory(masks_dir)
    create_directory(masked_images_dir)
    create_directory(low_conf_dir)
    
    # Create a log file to track processing details
    log_path = os.path.join(output_dir, "processing_log.txt")
    processed_count = 0
    low_conf_count = 0
    
    with open(log_path, 'w') as log_file:
        log_file.write(f"Processing {len(image_files)} images with threshold {threshold}\n")
        log_file.write("=" * 80 + "\n")
        
        for image_path in tqdm(image_files, desc="Processing images"):
            # Extract filename without extension
            filename = os.path.basename(image_path)
            base_filename, _ = os.path.splitext(filename)
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                log_file.write(f"ERROR: Could not read image {image_path}\n")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image in model
            model.set_image(image)
            
            # Generate mask using automatic segmentation (no points)
            masks, scores, _ = model.predict()
            
            # Log mask information
            log_file.write(f"Image: {filename}, Found {len(masks)} masks\n")
            if len(masks) > 0:
                log_file.write(f"  Scores: {[float(f'{s:.4f}') for s in scores]}\n")
                log_file.write(f"  Max score: {float(f'{max(scores):.4f}')}, Threshold: {threshold}\n")
            
            # If any mask found above threshold
            if len(masks) > 0 and max(scores) > threshold:
                # Get best mask
                best_idx = scores.argmax()
                mask = masks[best_idx].astype(bool)  # Ensure mask is boolean type
                
                # Save mask as binary .npy file in masks subdirectory
                mask_output_path = os.path.join(masks_dir, f"{base_filename}_mask.npy")
                np.save(mask_output_path, mask)
                
                # Extract only the masked region (where mask is True)
                masked_region = image.copy()
                # Create a black image where the mask is False
                masked_region[~mask] = 0
                
                # Save only the masked region of the image in masked_images subdirectory
                masked_output_path = os.path.join(masked_images_dir, f"{base_filename}_masked.png")
                cv2.imwrite(masked_output_path, cv2.cvtColor(masked_region, cv2.COLOR_RGB2BGR))
                
                processed_count += 1
                log_file.write(f"  [SUCCESS] Saved high-confidence mask and masked image\n")
            else:
                # If masks were found but below threshold, save the best one anyway in low_confidence folder
                if len(masks) > 0:
                    best_idx = scores.argmax()
                    mask = masks[best_idx].astype(bool)
                    
                    # Save mask to low confidence directory
                    low_conf_mask_path = os.path.join(low_conf_dir, f"{base_filename}_mask.npy")
                    np.save(low_conf_mask_path, mask)
                    
                    # Save visualization for debugging
                    masked_region = image.copy()
                    masked_region[~mask] = 0
                    low_conf_image_path = os.path.join(low_conf_dir, f"{base_filename}_masked.png")
                    cv2.imwrite(low_conf_image_path, cv2.COLOR_RGB2BGR)
                    
                    low_conf_count += 1
                    log_file.write(f"  [WARNING] Saved low-confidence mask (best score: {float(f'{max(scores):.4f}')})\n")
                else:
                    log_file.write(f"  [FAILED] No masks found\n")
            
            log_file.write("-" * 40 + "\n")
            
        # Write summary statistics
        log_file.write("\nSUMMARY\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Total images processed: {len(image_files)}\n")
        log_file.write(f"Images with high-confidence masks: {processed_count}\n")
        log_file.write(f"Images with low-confidence masks: {low_conf_count}\n")
        log_file.write(f"Images with no masks: {len(image_files) - processed_count - low_conf_count}\n")
    
    print(f"Processed {processed_count} images with high confidence")
    print(f"Found {low_conf_count} images with low confidence masks (below threshold {threshold})")
    print(f"See {log_path} for detailed processing information")

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

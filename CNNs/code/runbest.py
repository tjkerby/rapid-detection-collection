import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import shutil
from pathlib import Path
import time
from cnn_models import RiverClassifier
from torch.utils.data import Dataset, DataLoader

def print_gpu_info():
    # if torch.cuda.is_available():
    #     print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    #     print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    #     print(f"Current Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    pass

def load_model(model_type, model_path, num_classes):
    """
    Load a trained model based on the specified model type
    """
    if not torch.cuda.is_available():
        print("CUDA is not available! Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print("CUDA is available! Using GPU.")
        print_gpu_info()

    classifier = RiverClassifier(num_classes=num_classes, model_type=model_type)
    classifier.model = classifier.model.to(device)  # Explicitly move model to GPU
    
    state_dict = torch.load(model_path, map_location=device)
    classifier.model.load_state_dict(state_dict)
    classifier.model.eval()
    
    # Verify model is on GPU
    print(f"Model is on CUDA: {next(classifier.model.parameters()).is_cuda}")
    print_gpu_info()
    
    return classifier.model, device

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)  # Convert Path to string here

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def process_images(model, device, input_dir, output_dir, class_names, confidence_threshold=0.7, batch_size=8, num_workers=4):
    """
    Process images from input directory and sort them based on prediction confidence using batch processing
    """
    # Create transform for inference
    transform = transforms.Compose([
        transforms.Resize((1280, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "low_confidence"), exist_ok=True)
    
    # Collect all valid image paths
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for img_path in Path(input_dir).glob("**/*"):
        if img_path.suffix.lower() in valid_extensions:
            image_paths.append(img_path)
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)  # Disabled pin_memory
    
    # Process batches
    processed_count = 0
    total_images = len(image_paths)
    print(f"Found {total_images} images to process in batches of {batch_size}")
    
    for batch_images, batch_paths in dataloader:
        try:
            clear_gpu_memory()  # Clear GPU memory before each batch
            
            # Move batch to device
            batch_images = batch_images.to(device)
            
            # Perform inference
            with torch.no_grad():
                outputs = model(batch_images)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, dim=1)
                
                # Move results back to CPU and free GPU memory
                confidences = confidences.cpu().numpy()
                predicted_classes = predicted_classes.cpu().numpy()
                
                # Clear intermediate tensors
                del outputs, probabilities
                batch_images = batch_images.cpu()
                del batch_images
                clear_gpu_memory()
            
            # Process each result in the batch
            for i in range(len(batch_paths)):
                img_path = Path(batch_paths[i])  # Convert string back to Path object
                confidence = confidences[i]
                predicted_class = predicted_classes[i]
                
                # Determine destination folder based on confidence
                if confidence >= confidence_threshold:
                    dest_folder = os.path.join(output_dir, class_names[predicted_class])
                else:
                    dest_folder = os.path.join(output_dir, "low_confidence")
                
                # Copy the image to destination folder
                dest_file = os.path.join(dest_folder, f"{img_path.stem}_conf{confidence:.2f}{img_path.suffix}")
                shutil.copy2(img_path, dest_file)
            
            processed_count += len(batch_paths)
            if processed_count % 50 == 0 or processed_count == total_images:
                print(f"Processed {processed_count}/{total_images} images...")
                print_gpu_info()
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # Try to recover by clearing memory
            clear_gpu_memory()
            continue
    
    print(f"Finished processing {processed_count} images.")
    return processed_count

def main():
    start_time = time.time()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run trained model to sort images based on prediction confidence')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['simple', 'resnet', 'efficientnet', 'resnetpre', 'efficientnetpre'],
                        help='Type of model to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save sorted images')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Confidence threshold for class assignment (default: 0.7)')
    parser.add_argument('--class_names', type=str, nargs='+', required=True,
                        help='Names of the classes in the model')
    parser.add_argument('--batch_size', type=int, default=16,  # Reduced default batch size
                        help='Batch size for processing images (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading (default: 4)')
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model_type, args.model_path, len(args.class_names))
    print(f"Model loaded: {args.model_type}")
    print(f"Using device: {device}")
    
    # Process and sort images
    print(f"Processing images from {args.input_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    count = process_images(model, device, args.input_dir, args.output_dir, 
                         args.class_names, args.confidence_threshold,
                         args.batch_size, args.num_workers)
    
    elapsed_time = time.time() - start_time
    print(f"Sorted {count} images into {len(args.class_names)} class folders")
    print(f"Images with confidence below {args.confidence_threshold} were placed in 'low_confidence' folder")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

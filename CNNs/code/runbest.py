import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import shutil
from pathlib import Path
from cnn_models import RiverClassifier

def load_model(model_type, model_path, num_classes):
    """
    Load a trained model based on the specified model type
    """
    classifier = RiverClassifier(num_classes=num_classes, model_type=model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.model.load_state_dict(torch.load(model_path, map_location=device))
    classifier.model.eval()
    return classifier.model, device

def process_images(model, device, input_dir, output_dir, class_names, confidence_threshold=0.7):
    """
    Process images from input directory and sort them based on prediction confidence
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
    
    # Process each image
    processed_count = 0
    for img_path in Path(input_dir).glob("**/*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    confidence, predicted_class = torch.max(probabilities, 0)
                    
                    # Determine destination folder based on confidence
                    if confidence >= confidence_threshold:
                        dest_folder = os.path.join(output_dir, class_names[predicted_class])
                    else:
                        dest_folder = os.path.join(output_dir, "low_confidence")
                
                # Copy the image to destination folder
                dest_file = os.path.join(dest_folder, f"{img_path.stem}_conf{confidence:.2f}{img_path.suffix}")
                shutil.copy2(img_path, dest_file)
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} images...")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Finished processing {processed_count} images.")
    return processed_count

def main():
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
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model_type, args.model_path, len(args.class_names))
    print(f"Model loaded: {args.model_type}")
    print(f"Using device: {device}")
    
    # Process and sort images
    print(f"Processing images from {args.input_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    count = process_images(model, device, args.input_dir, args.output_dir, 
                         args.class_names, args.confidence_threshold)
    
    print(f"Sorted {count} images into {len(args.class_names)} class folders")
    print(f"Images with confidence below {args.confidence_threshold} were placed in 'low_confidence' folder")

if __name__ == "__main__":
    main()

import os
import shutil
import pandas as pd

def setup_directories():
    """Create all necessary output directories"""
    directories = ["./rapids", "./norapids", "./RapidsMask", "./noRapidsMask"]
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)

def process_entry(row, base_paths):
    """Process a single entry from the CSV"""
    try:
        # Skip if no rapids_class
        if pd.isna(row['rapid_class']):
            return
        
        image_name = row['image']  # Just use the image column
        rapids_class = int(row['rapid_class'])
        has_mask = int(row['mask']) if 'mask' in row else 0
        
        # Determine destination folders based on rapids_class
        img_dest_folder = "./rapids" if rapids_class == 1 else "./norapids"
        mask_dest_folder = "./RapidsMask" if rapids_class == 1 else "./noRapidsMask"
        
        # Process image
        source_image = os.path.join(base_paths['images'], f"{image_name}.png")
        dest_image = os.path.join(img_dest_folder, f"{image_name}.png")
        
        if os.path.exists(source_image):
            shutil.copy2(source_image, dest_image)
            print(f"Copied {image_name}.png to {img_dest_folder}")
        else:
            print(f"Warning: Source image {source_image} not found")
        
        # Process mask if it exists
        if has_mask == 1:
            source_mask = os.path.join(base_paths['masks'], f"{image_name}.npy")
            dest_mask = os.path.join(mask_dest_folder, f"{image_name}.npy")
            
            if os.path.exists(source_mask):
                shutil.copy2(source_mask, dest_mask)
                print(f"Copied {image_name}.npy to {mask_dest_folder}")
            else:
                print(f"Warning: Source mask {source_mask} not found")
                
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")

def main():
    # Base paths
    base_paths = {
        'images': "C:/Users/kaden/Box/STAT5810/rapids/data/images",
        'masks': "C:/Users/kaden/Box/STAT5810/rapids/data/masks"
    }
    
    # Create output directories
    setup_directories()
    
    # Read CSV file
    try:
        csv_path = "C:/Users/kaden/Box/STAT5810/rapids/data/meta_csv/meta.csv"
        df = pd.read_csv(csv_path)
        
        # Verify required columns - removed 'name'
        required_columns = ['image', 'rapid_class']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            return
        
        # Process each row
        for _, row in df.iterrows():
            process_entry(row, base_paths)
            
        print("Processing complete!")
        
    except FileNotFoundError:
        print(f"Error: CSV file not found")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
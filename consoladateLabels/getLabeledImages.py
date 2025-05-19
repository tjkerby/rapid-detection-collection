import os
import shutil
import pandas as pd
import threading
import time

def safe_copy(src, dst, timeout_secs=10, max_retries=3):
    def copy_operation(success_event):
        try:
            shutil.copy2(src, dst)
            success_event.set()
        except Exception as e:
            print(f"\nThread error copying {src}: {e}")

    for attempt in range(max_retries):
        success_event = threading.Event()
        thread = threading.Thread(target=copy_operation, args=(success_event,))
        thread.daemon = True  # Allow program to exit even if thread is stuck
        
        print(f"\nAttempt {attempt + 1}/{max_retries} copying {src}")
        thread.start()
        
        if success_event.wait(timeout=timeout_secs):
            print(f"\nSuccessfully copied {src}")
            return True
            
        print(f"\nTimeout on attempt {attempt + 1} for {src}")
        time.sleep(2)  # Longer wait between retries
        
    print(f"\nGiving up on {src} after {max_retries} attempts")
    return False

def setup_directories():
    """Create all necessary output directories"""
    directories = [
        "./rapids", "./norapids",
        "./RapidsMask", "./noRapidsMask",
        "./uhj", "./nouhj",
        "./uhj_masked", "./nouhj_masked"    # added UHJ mask dirs
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

def process_entry(row, base_paths, current_count, total_count):
    """Process a single entry from the CSV"""
    try:
        image_name = row['image']
        print(f"\rProcessing {current_count}/{total_count}: {image_name}", end="")
        
        source_image = os.path.join(base_paths['images'], f"{image_name}.png")
        source_mask  = os.path.join(base_paths['masks'],  f"{image_name}.npy")

        success = True  # Track if all operations succeeded
        
        # --- Rapids split ---
        if not pd.isna(row['rapid_class']):
            rapids_class = int(row['rapid_class'])
            dest_folder = "./rapids" if rapids_class == 1 else "./norapids"
            if os.path.exists(source_image):
                success &= safe_copy(source_image, os.path.join(dest_folder, f"{image_name}.png"))

        # --- Rapids mask ---
        if 'mask' in row and not pd.isna(row['mask']) and not pd.isna(row['rapid_class']) and int(row['mask']) == 1:
            mask_folder = "./RapidsMask" if int(row['rapid_class']) == 1 else "./noRapidsMask"
            if os.path.exists(source_mask):
                success &= safe_copy(source_mask, os.path.join(mask_folder, f"{image_name}.npy"))

        # --- UHJ split (only if properly labeled) ---
        if ('uhj_class' in row and not pd.isna(row['uhj_class']) and 
            'uhj_labeled_by' in row and not pd.isna(row['uhj_labeled_by'])):
            uhj_val = float(row['uhj_class'])
            
            # For nouhj: require uhj_class=0 AND (rapid_class=1 OR rapid_class=NA)
            is_nouhj = (uhj_val == 0 and 
                       ('rapid_class' not in row or pd.isna(row['rapid_class']) or int(row['rapid_class']) == 1))
            
            uhj_folder = "./uhj" if uhj_val >= 0.5 else (
                "./nouhj" if is_nouhj else None
            )
            
            if uhj_folder and os.path.exists(source_image):
                success &= safe_copy(source_image, os.path.join(uhj_folder, f"{image_name}.png"))
                
                # --- UHJ masked split ---
                if 'mask' in row and not pd.isna(row['mask']) and int(row['mask']) == 1 and os.path.exists(source_mask):
                    uhj_mask_folder = "./uhj_masked" if uhj_val >= 0.5 else "./nouhj_masked"
                    success &= safe_copy(source_mask, os.path.join(uhj_mask_folder, f"{image_name}.npy"))
        
        if success:
            print(f"\rProcessed {image_name} successfully.")
        else:
            print(f"\rPartially processed {image_name} with some failures.")

    except Exception as e:
        print(f"\nError processing {image_name}: {e}")

def main():
    base_paths = {
        'images': "C:/Users/kaden/Box/STAT5810/rapids/data/images",
        'masks' : "C:/Users/kaden/Box/STAT5810/rapids/data/masks"
    }
    setup_directories()

    csv_path = "C:/Users/kaden/Box/STAT5810/rapids/data/meta_csv/meta.csv"
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError:
        print("CSV not found"); return
    except pd.errors.EmptyDataError:
        print("CSV is empty"); return

    required = ['image','rapid_class']
    if not all(c in df.columns for c in required):
        print(f"Missing columns: {required}"); return

    total_rows = len(df)
    for idx, row in df.iterrows():
        process_entry(row, base_paths, idx + 1, total_rows)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
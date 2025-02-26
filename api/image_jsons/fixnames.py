import os

def main():
    # Get the directory where the script resides
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Iterate over all items in the directory
    for filename in os.listdir(script_dir):
        file_path = os.path.join(script_dir, filename)
        
        # Process only files
        if os.path.isfile(file_path):
            # Split the filename into base and extension
            base, ext = os.path.splitext(filename)
            
            # Check if the base ends with "_s1"
            if base.endswith("_s1"):
                # Remove the "_s1" part (3 characters)
                new_base = base[:-3]
                new_name = new_base + ext
                new_file_path = os.path.join(script_dir, new_name)
                
                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_name}'")

if __name__ == "__main__":
    main()
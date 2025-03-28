import os
import json
import shutil
def get_entries_in_directory(directory_path):

    
    # Hardcoded directory path
    directory_path = "C:\\Users\\kaden\\Box\\STAT5810\\rapids\\data\\masks"  # Change this to your actual directory path
    
    try:
        # Get all entries in the directory
        entries = os.listdir(directory_path)
        return entries
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied to access '{directory_path}'.")
        return []

if __name__ == "__main__":
    entries = get_entries_in_directory(None)  # None since we're using the hardcoded path
    strippedEntries = [entry.split('.npy')[0] for entry in entries]
    
    # Create output directories if they don't exist
    if not os.path.exists("./rapids"):
        os.makedirs("./rapids")
    if not os.path.exists("./norapids"):
        os.makedirs("./norapids")
    
    for entry in strippedEntries:
        try:
            # Correctly open the json file for this entry
            json_file_path = os.path.join("C:\\Users\\kaden\\Box\\STAT5810\\rapids\\data\\jsons", f"{entry}.json")
            with open(json_file_path, 'r') as jsonFile:
                jsonDict = json.load(jsonFile)
                
                # Get the source image path
                source_image_path = os.path.join("C:\\Users\\kaden\\Box\\STAT5810\\rapids\\data\\images", f"{entry}.png")
                
                if jsonDict["rapid_class"] == 1:
                    # Get the destination path for rapids
                    dest_image_path = os.path.join("./rapids", f"{entry}.png")
                    
                    # Copy the file if it exists
                    if os.path.exists(source_image_path):
                        shutil.copy2(source_image_path, dest_image_path)
                        print(f"Copied {entry}.png to ./rapids folder")
                    else:
                        print(f"Warning: Source image {source_image_path} not found")
                    
                elif jsonDict["rapid_class"] == 0:
                    # Get the destination path for norapids
                    dest_image_path = os.path.join("./norapids", f"{entry}.png")
                    
                    # Copy the file if it exists
                    if os.path.exists(source_image_path):
                        shutil.copy2(source_image_path, dest_image_path)
                        print(f"Copied {entry}.png to ./norapids folder")
                    else:
                        print(f"Warning: Source image {source_image_path} not found")
        except FileNotFoundError:
            print(f"Warning: JSON file for {entry} not found")
        except PermissionError:
            print(f"Error: Permission denied when trying to access JSON file for {entry}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file for {entry}")
        except Exception as e:
            print(f"Error processing {entry}: {str(e)}")


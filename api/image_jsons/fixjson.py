import os
import json

def fix_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                continue

            modified = False

            # If JSON is a dictionary with an "image" key
            if isinstance(data, dict) and "image" in data and isinstance(data["image"], str):
                new_value = data["image"].replace("\\", "/")
                if new_value != data["image"]:
                    data["image"] = new_value
                    modified = True

            # Alternatively, if JSON is a list of dictionaries
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "image" in item and isinstance(item["image"], str):
                        new_value = item["image"].replace("\\", "/")
                        if new_value != item["image"]:
                            item["image"] = new_value
                            modified = True

            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                print(f"Updated {filename}")

if __name__ == "__main__":
    # Get the directory where the script is located
    current_directory = os.path.dirname(os.path.realpath(__file__))
    fix_json_files(current_directory)
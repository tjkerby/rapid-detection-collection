import json
import os
import sys
from pull_image import get_satellite_image

def process_coordinates(json_file, output_dir="satellite_images", limit=None, zoom=18):
    """
    Process coordinates from JSON file and download satellite images
    Args:
        json_file (str): Path to JSON file containing coordinates
        output_dir (str): Directory to store downloaded images
        limit (int): Optional limit on number of API requests
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and parse JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    request_count = 0
    # Process each feature in the collection
    for feature in data['features']:
        river_name = "unnamed" if 'name' not in feature['properties'] else feature['properties']['name'].replace(" ", "_")
        coordinates = feature['geometry']['coordinates']
        
        # Process each coordinate pair
        for coord in coordinates:
            if limit and request_count >= limit:
                print(f"Request limit of {limit} reached. Stopping.")
                return
                
            longitude, latitude = coord[0], coord[1]
            
            # Create filename using river name and coordinates with zoom at end
            filename = f"{river_name}_{longitude}_{latitude}_z{zoom}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Skip if image already exists
            if os.path.exists(filepath):
                print(f"Skipping existing image: {filename}")
                continue
            
            try:
                # Call the satellite image API
                get_satellite_image(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=zoom,  # Adjust zoom level as needed
                    size="1280x1280",
                    scale=4,
                    output_path=filepath
                )
                request_count += 1
                print(f"Downloaded: {filename} ({request_count} requests made)")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python pull_images.py <json_file> [output_dir] [limit]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'satellite_images'
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    process_coordinates(json_file, output_dir, limit,19)

if __name__ == "__main__":
    main()

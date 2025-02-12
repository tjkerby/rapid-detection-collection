import json
import os
import sys
from pull_image import get_satellite_image
import argparse 
def process_coordinates(json_file, output_dir="satellite_images", limit=None, zoom=18,start=None,end=None):
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
    hasStarted = start is None
    for feature in data['features']:
        
        river_name = "unnamed" if 'name' not in feature['properties'] else feature['properties']['name'].replace(" ", "_")
        if river_name == start:
            hasStarted = True
        if hasStarted:
            coordinates = feature['geometry']['coordinates']
            
            # Process each coordinate pair
            for coord in coordinates:
                if limit and request_count >= limit:
                    print(f"Request limit of {limit} reached. Stopping.")
                    return
                    
                print(f"Processing: {river_name}")
                print(f"Coordinates: {coord}")
                if isinstance(coord, list):
                    longitude, latitude = coord[0], coord[1]
                else:
                    longitude, latitude = coordinates[0], coordinates[1]
                
                # Create filename using river name and coordinates with zoom at end
                filename = f"{river_name}_{longitude}_{latitude}_z{zoom}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Skip if image already exists
                if os.path.exists(filepath) or os.path.exists("image_jsons/"+filename.replace("png","json")):
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
                    filenameJson = f"{river_name}_{longitude}_{latitude}_z{zoom}.json"
                    # filepathJson = os.path.join(output_dir, filename)
                    dict = {
                        "name": river_name,
                        "longitude": longitude,
                        "latitude": latitude,
                        "zoom": zoom,
                        "image": filepath,
                        "class": "",
                        "map": ""
                    }

                    with open("./image_jsons/"+filenameJson, 'w') as f:
                        json.dump(dict, f, indent=4)
                except Exception as e:
                    print(f"Error downloading {filename}: {str(e)}")
        if river_name == end:
            break

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python pull_images.py <json_file> [output_dir] [limit]")
    #     sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Process coordinates from JSON file and download satellite images')
    parser.add_argument('--json_file', help='Path to JSON file containing coordinates')
    parser.add_argument('--output_dir', default='images', help='Directory to store downloaded images')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of API requests')
    parser.add_argument('--zoom', type=int, default=18, help='Zoom level for satellite images')
    parser.add_argument('--start', help='River name to start processing from')
    parser.add_argument('--end', help='River name to end processing at')
    
    args = parser.parse_args()
    
    json_file = args.json_file
    output_dir = args.output_dir
    limit = args.limit
    zoom = args.zoom
    start = args.start
    end = args.end
    process_coordinates(json_file, output_dir, limit,zoom,start,end)

if __name__ == "__main__":
    main()

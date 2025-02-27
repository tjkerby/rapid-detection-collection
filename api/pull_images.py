import json
import os
import sys
import csv  # added for CSV support
from pull_image import get_satellite_image
import argparse
from dotenv import load_dotenv

def process_coordinates(json_file, output_dir, limit=None, zoom=18, start=None, end='thereisnoend!'):
    """
    Process coordinates from JSON file and download satellite images
    Args:
        json_file (str): Path to JSON file containing coordinates
        output_dir (str): Directory to store downloaded images
        limit (int): Optional limit on number of API requests
    """
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return
    
    # Set json output directory from the env variable 'DATA_PATH'
    env_path = os.environ.get('DATA_PATH')
    if not env_path:
        env_path = "c:/Users/kaden/Independent-Study/rapid-detection-collection"
    json_dir = os.path.join(env_path, "jsons")
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    request_count = 0

    if json_file.lower().endswith(".csv"):
        # Process CSV file input
        base_name = os.path.basename(json_file)
        river_name = os.path.splitext(base_name)[0].replace(" ", "_")
        with open(json_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and request_count >= limit:
                    print(f"Request limit of {limit} reached. Stopping.")
                    return
                try:
                    latitude = float(row['lat'])
                    longitude = float(row['long'])
                except Exception as e:
                    print(f"Error reading row {row}: {str(e)}")
                    continue
                print(f"Processing: {river_name}")
                filename = f"{river_name}_{longitude}_{latitude}_z{zoom}.png"
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath) or os.path.exists("image_jsons/"+filename.replace("png","json")):
                    print(f"Skipping existing image: {filename}")
                    continue
                try:
                    get_satellite_image(
                        latitude=latitude,
                        longitude=longitude,
                        zoom=zoom,
                        size="1280x1280",
                        scale=4,
                        output_path=filepath
                    )
                    request_count += 1
                    print(f"Downloaded: {filename} ({request_count} requests made)")
                    filenameJson = f"{river_name}_{longitude}_{latitude}_z{zoom}.json"
                    data_dict = {
                        "name": river_name,
                        "longitude": longitude,
                        "latitude": latitude,
                        "zoom": zoom,
                        "image": filepath.replace("\\","/"),
                        "rapid_class": "",
                        "uhj_class": "",
                        "map": ""
                    }
                    with open(os.path.join(json_dir, filenameJson), 'w') as jf:
                        json.dump(data_dict, jf, indent=4)
                except Exception as e:
                    print(f"Error downloading {filename}: {str(e)}")
        return
    else:
        # Existing JSON processing branch
        with open(json_file, 'r') as f:
            data = json.load(f)
        hasStarted = start is None
        for feature in data['features']:
            river_name = "unnamed" if 'name' not in feature['properties'] else feature['properties']['name'].replace(" ", "_")
            if river_name == start:
                hasStarted = True
            if hasStarted:
                coordinates = feature['geometry']['coordinates']
                # Process each coordinate pair
                for coords in coordinates:
                    if feature['geometry']['type'] == 'Polygon':
                        if len(coords) > 0:
                            avg_lon = sum(point[0] for point in coords) / len(coords)
                            avg_lat = sum(point[1] for point in coords) / len(coords)
                            coord = [avg_lon, avg_lat]
                        else:
                            coord = coords
                    else:
                        coord = coords
                    if limit and request_count >= limit:
                        print(f"Request limit of {limit} reached. Stopping.")
                        return
                    print(f"Processing: {river_name}")
                    print(f"Coordinates: {coord}")
                    if isinstance(coord, list):
                        longitude, latitude = coord[0], coord[1]
                    else:
                        longitude, latitude = coordinates[0], coordinates[1]
                    filename = f"{river_name}_{longitude}_{latitude}_z{zoom}.png"
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath) or os.path.exists("image_jsons/"+filename.replace("png","json")):
                        print(f"Skipping existing image: {filename}")
                        continue
                    try:
                        get_satellite_image(
                            latitude=latitude,
                            longitude=longitude,
                            zoom=zoom,
                            size="1280x1280",
                            scale=4,
                            output_path=filepath
                        )
                        request_count += 1
                        print(f"Downloaded: {filename} ({request_count} requests made)")
                        filenameJson = f"{river_name}_{longitude}_{latitude}_z{zoom}.json"
                        data_dict = {
                            "name": river_name,
                            "longitude": longitude,
                            "latitude": latitude,
                            "zoom": zoom,
                            "image": filepath.replace("\\","/"),
                            "rapid_class": "",
                            "uhj_class": "",
                            "map": ""
                        }#
                        with open(os.path.join(json_dir, filenameJson), 'w') as jf:
                            json.dump(data_dict, jf, indent=4)
                    except Exception as e:
                        print(f"Error downloading {filename}: {str(e)}")
            if river_name == end.replace(" ", "_"):
                return

def main():
    parser = argparse.ArgumentParser(description='Process coordinates from JSON/CSV file and download satellite images')
    parser.add_argument('--json_file', help='Path to JSON or CSV file containing coordinates')
    parser.add_argument('--output_dir', default='images', help='Directory to store downloaded images')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of API requests')
    parser.add_argument('--zoom', type=int, default=18, help='Zoom level for satellite images')
    parser.add_argument('--start', help='River name to start processing from')
    parser.add_argument('--end', help='River name to end processing at', default='thereisnoend!')
    
    args = parser.parse_args()
    load_dotenv()
    env_path = os.environ.get('DATA_PATH')
    if not env_path:
        env_path = "c:/Users/kaden/Independent-Study/rapid-detection-collection"
    full_output_dir = os.path.join(env_path, args.output_dir)
    process_coordinates(args.json_file, full_output_dir, args.limit, args.zoom, args.start, args.end)

if __name__ == "__main__":
    main()

import json
import os

def write_image_json(filepath):

    filepath = filepath.split("_")
    river_name = filepath[0]
    longitude = filepath[1]
    float_longitude = float(longitude)
    latitude = filepath[2]
    float_latitude = float(latitude)
    zoom = 18
    image = "".join(["images/", river_name, "_", longitude, "_", latitude, "_z18.png"])

    filenameJson = f"{river_name}_{longitude}_{latitude}_z{zoom}_s{1}.json"
    dict = {
        "name": river_name,
        "longitude": float_longitude,
        "latitude": float_latitude,
        "zoom": zoom,
        "image": image,
        "class": "",
        "map": ""
    }

    with open("api/image_jsons/"+filenameJson, 'w') as f:
        json.dump(dict, f, indent=4)

def main():

    image_dir = "C:/Users/nbrim/Desktop/STAT 5810 ML/rapids_images"

    image_paths = os.listdir(image_dir)
    
    for i in range(len(image_paths)):
        write_image_json(image_paths[i])


if __name__ == "__main__":
    main()


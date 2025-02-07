import os
import requests
from dotenv import load_dotenv

def get_satellite_image(latitude, longitude, zoom, size="1280x1280", scale=2):
    """
    Fetch satellite image from Google Maps Static API
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        zoom (int): Zoom level (0-21, where 21 is closest)
        size (str): Image dimensions in pixels (width x height), max 1280x1280
        scale (int): Image scale/resolution multiplier (1, 2, or 4)
    """
    # Try loading from .env file first
    load_dotenv()
    # Check for API key in both .env and system environment variables
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        raise ValueError("Google Maps API key not found. Please set GOOGLE_MAPS_API_KEY in system environment variables or .env file")

    # Construct the URL for Google Maps Static API
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        'center': f'{latitude},{longitude}',
        'zoom': zoom,
        'size': size,
        'maptype': 'satellite',
        'key': api_key,
        'scale': scale
    }

    # Make the request
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Create filename using coordinates
        filename = f"satellite_{latitude}_{longitude}_{zoom}.png"
        
        # Save the image
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {filename}")
    else:
        print(f"Error: Unable to fetch image. Status code: {response.status_code}")

if __name__ == "__main__":
    # Example usage
    latitude = 41.7306162
    longitude = -111.8154619
    zoom = 20 # Adjust zoom level as needed
    get_satellite_image(latitude, longitude, zoom, scale=2)
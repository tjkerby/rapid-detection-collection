from fastkml import kml
import json
from shapely.geometry import mapping
import os

def kml_to_json(kml_path, json_path=None):
    """
    Convert a KML file to JSON format
    
    Args:
        kml_path (str): Path to input KML file
        json_path (str): Path to output JSON file (optional)
    
    Returns:
        dict: JSON-compatible dictionary of KML contents
    """
    # Read KML file
    with open(kml_path, 'rb') as kml_file:
        kml_data = kml_file.read()
    
    # Parse KML
    k = kml.KML()
    k.from_string(kml_data)
    
    # Initialize result dictionary
    result = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    # Extract features from KML
    def extract_features(element):
        if hasattr(element, 'features'):
            for feature in element.features:  # revert from element.features()
                if hasattr(feature, 'geometry'):
                    feature_dict = {
                        'type': 'Feature',
                        'properties': {
                            'name': feature.name if hasattr(feature, 'name') else '',
                            'description': feature.description if hasattr(feature, 'description') else ''
                        },
                        'geometry': mapping(feature.geometry) if feature.geometry else None
                    }
                    result['features'].append(feature_dict)
                
                # Recursively process any nested features
                extract_features(feature)

    # Process all features
    for feature in k.features:  # revert from k.features()
        extract_features(feature)

    # Write to JSON file if path is provided
    if json_path:
        with open(json_path, 'w') as json_file:
            json.dump(result, json_file, indent=2)
    
    return result

if __name__ == '__main__': 
    # Example usage
    input_kml = 'kml_test.kml'  # Replace with your KML file path
    output_json = 'output.json'  # Replace with desired output path
    
    try:
        converted_data = kml_to_json(input_kml, output_json)
        print(f"Successfully converted {input_kml} to {output_json}")
    except Exception as e:
        print(f"Error converting KML to JSON: {str(e)}")
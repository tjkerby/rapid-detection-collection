# Rapid Detection Collection

A tool for collecting and cataloging satellite imagery, particularly designed for detecting and classifying rapids in rivers using Google Maps Static API.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides tools to download high-resolution satellite imagery from Google Maps Static API using coordinates from JSON or CSV files. It's primarily designed for collecting images of river rapids for classification and analysis.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rapid-detection-collection.git
cd rapid-detection-collection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root directory.
2. Add your Google Maps API key and data directory:
```
GOOGLE_MAPS_API_KEY=your_api_key_here
DATA_PATH=/path/to/your/data/directory
```

To obtain a Google Maps API key:
1. Visit the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the "Maps Static API"
4. Create credentials for an API key

## Usage

### Basic Usage

To download satellite images using a JSON coordinate file:

```bash
python api/pull_images.py --json_file path/to/coordinates.json --output_dir images --zoom 18
```

To download satellite images using a CSV coordinate file:

```bash
python api/pull_images.py --json_file path/to/coordinates.csv --output_dir images --zoom 18
```

### Command Line Arguments

- `--json_file`: Path to JSON or CSV file containing coordinates
- `--output_dir`: Directory to store downloaded images (default: 'images')
- `--limit`: Optional limit on number of API requests
- `--zoom`: Zoom level for satellite images (default: 18, range: 0-21)
- `--start`: River name to start processing from
- `--end`: River name to end processing at

### Example CSV Format

The CSV file should have `lat` and `long` columns:

```
lat,long
44.07510219825334,-71.68399351583908
```

### Example JSON Structure

```json
{
  "features": [
    {
      "properties": {
        "name": "River Name"
      },
      "geometry": {
        "type": "Point|LineString|Polygon",
        "coordinates": [[-71.68399, 44.07510]]
      }
    }
  ]
}
```

## Features

- Download high-resolution satellite imagery from Google Maps
- Process coordinate data from JSON or CSV files
- Store images with metadata in an organized directory structure
- Skip existing images to avoid duplicate downloads
- Create metadata JSON files for each downloaded image
- Configurable image parameters (zoom level, size, scale)
- Process polygon coordinates by using the centroid

## File Structure

```
rapid-detection-collection/
├── api/
│   ├── pull_image.py      # Core function for downloading single images
│   ├── pull_images.py     # Script to process multiple coordinates
│   └── jsons/             # Generated metadata JSON files
├── images/                # Downloaded satellite images
└── .env                   # Environment variables
```


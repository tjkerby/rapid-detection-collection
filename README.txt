# Rapid Detection Collection

A tool for collecting and cataloging satellite imagery, particularly designed for detecting and classifying rapids in rivers using Google Maps Static API.

## Overview

This project provides tools to download high-resolution satellite imagery from Google Maps Static API using coordinates from JSON or CSV files. It's primarily designed for collecting images of river rapids for classification and analysis.

## Classifier

The Python code for training the rapids classification models was run using the uv package manager by Astral (https://docs.astral.sh/uv/). Once uv is installed, run the following commands in the terminal to setup a Python environment with all packages required to run the model training code.

1. Change the working directory to the classifier folder in the rapid-detection-collection code archive

2. Initialize a uv virtual environment using uv venv --python 3.12.3

3. Install required dependencies using uv pip install -r requirements.txt

Before training the rapids classifier, edit the file paths in rapids_train.py with the local machine paths for the rapids image dataset, the rapids class labels, and an output file directory. Select a model log name to identify the output files, including model weights, validation metric plots, and test dataset metrics. Then run the script using 

## API

The code for downloading images from the Google Maps Static API is organized in an R project under the api directory in the code archive. To retrieve additional images from the API, open the R project in RStudio and edit code/par_script.R to point to a CSV file containing fields for a river or watershed identifier and the latitude and longitude coordinates of the target locations. To enable the Maps Static API, follow the steps below. 

1. Create a `.env` file in the R project directory
2. Add a Google Maps API key and secret like so

```
GOOGLE_MAPS_API_KEY=api_key_here
GOOGLE_MAPS_API_SECRET=secret_here
```

To obtain a Google Maps API key:
1. Visit the Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project
3. Enable the "Maps Static API"
4. Create credentials for an API key

Once an API key has been created, a secret can be obtained from the Keys & Credentials section of the Maps Platform page (https://console.cloud.google.com/google/maps-apis).

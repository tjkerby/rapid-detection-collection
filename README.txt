# Rapid Detection Collection

A set of tools for downloading and classifying satellite imagery of river rapids

## API

Directories and files: 

api
   - rapids.api: An R project containing the code for downloading satellite imagery from the Google Maps Static API.
      - code
         - par_script.R: The driver R script used for accessing the API.
         - pull_maps_image.R: An R file containing a function for pulling and locally writing a single Maps Static API image.
         - sign.R: An R file containing a function implementing API key signatures using a cryptographic hashing algorithm. 
         - kml_to_csv.R: An R script used to extract coordinates for the locations in the known rapids datasets using the xml2 package.
         - kml_helpers.csv: An R file containing helper functions for processing the known rapids datasets.
      - kmls
         - NHDArea_rapids_slope.kml: A Keyhole Markup Language (KML) file containing spatial features for known rapids locations from the National Hydrography Dataset (NHD).
         - OSMrapidsAll.kml: A KML file containing spatial features for known rapids locations from OpenStreetMap (OSM).

To retrieve additional images from the API, follow these steps. 

1. Open the R project in RStudio

2. Open code/par_script.R and edit the `flowline_pts_csv_path` variable to point to a CSV file of target locations containing the following fields:
   - title: A river or watershed identifier for the target location.
   - latitude: The latitude coordinate of the target location.
   - longitude: The latitude coordinate of the target location.

3. Create a .env file in the R project directory with the following structure:

```
GOOGLE_MAPS_API_KEY=api_key_here
GOOGLE_MAPS_API_SECRET=secret_here
```

4. (Optional) To run the API using multiple cores, change the `workers` argument in the `future::plan` call. To see how many cores are on the local machine, run parallel::detectCores() in the R console. Note that general advice is to use fewer cores than are available to avoid conflicting with other processes.

5. Run code/par_script.R interactively to call the API and download Maps imagery for the target locations.

To obtain a Google Maps API key:
1. Visit the Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project
3. Enable the "Maps Static API"
4. Create credentials for an API key

Once an API key has been created, a secret can be obtained from the Keys & Credentials section of the Maps Platform page (https://console.cloud.google.com/google/maps-apis).

## Annotation

The annotation directory provides an interactive manual annotation system for creating ground truth data for river feature detection and rapid identification. The system combines segmentation mask creation with rapid classification workflows using SAM2 model integration.

Directories and files:
annotation
   - label.py: The core labeling engine providing an interactive annotation interface for manually labeling river images with segmentation masks and rapid classifications, supporting multiple annotation modes (mask-only, rapid-only, combined, and UHJ classification).
   - RapidsImage.py: A class that handles interactive image display, mouse-based annotation, and real-time mask visualization with SAM2 model integration for point-based segmentation assistance.
   - segmentation.py: The main entry point providing a menu-driven interface for selecting annotation workflows, combining SAM2 model integration with multiple annotation modes and comprehensive metadata tracking.
   - select_device.py: A utility module for automatic device selection and optimization configuration for PyTorch models, specifically optimized for SAM2 inference with CUDA/CPU detection and performance tuning.

Key Features:
- Interactive point-based annotation with SAM2 model assistance
- Multiple annotation workflows: segmentation masks, rapid classification, standing wave detection
- Real-time mask visualization with color overlays and mouse controls
- Progress tracking and resumable annotation sessions
- Automatic metadata and timestamp logging with quality control prompts

Setup Requirements:
Before running, create a '.user.json' file in the project root with:
```
{
    "user": "annotator_name",
    "metadata": "path/to/metadata.csv",
    "image_folder": "path/to/images/",
    "npy_folder": "path/to/output/masks/",
    "SAM2_CHECKPOINT_FOLDER": "path/to/sam2/checkpoints/"
}
```

To start the annotation process:
1. Navigate to the annotation directory
2. Run: `python segmentation.py`
3. Select your desired annotation workflow from the menu-driven interface

Annotation Controls:
- Left Click: Add positive points (include in segmentation)
- Right Click: Add negative points (exclude from segmentation)
- 't': Mark entire image as river
- 'f': Mark entire image as non-river
- 'z': Remove last point
- 'q': Save and continue
- '0'/'1': Binary classification for rapids
- '5': Maybe/uncertain for UHJ classification

Output:
- Updated CSV metadata with labels, timestamps, and annotator information
- Binary mask files (.npy format) for segmentation annotations
- Detailed annotation provenance and validation prompts

## Segmentation

The maskingTool directory provides automated river segmentation capabilities using fine-tuned SAM2 models for batch processing of river images. This tool generates predicted segmentation masks and evaluates model performance for large-scale river detection tasks.

Directories and files:
maskingTool
   - maskimages.py: Automatically generates predicted segmentation masks of rivers for a batch of images using a fine-tuned SAM2 model, with confidence-based filtering and organized output structure.
   - finetuning.ipynb: A Jupyter notebook that fine-tunes the SAM2 model on river images to create a specialized model for river channel segmentation with training, validation, and test loops.
   - predict_new.ipynb: A notebook for evaluating fine-tuned models, calculating IoU scores, determining optimal confidence thresholds, and visualizing prediction results on validation and test sets.
   - requirements.txt: Python package dependencies for the segmentation tools.

Key Features:
- Batch processing of images 
- Automated river detection using fine-tuned SAM2 model
- Confidence-based filtering with adjustable thresholds
- Model fine-tuning and evaluation workflows
- Intersection over Union (IoU) scoring for accuracy assessment
- Organized output with separate folders for different confidence levels

To run automated segmentation:
1. Navigate to the maskingTool directory
2. Run: `python maskimages.py --input_dir /path/to/images --output_dir /path/to/output`
3. Optional parameters: `--threshold 0.1 --checkpoint_dir ../checkpoints`

For model fine-tuning and evaluation:
1. Open `finetuning.ipynb` to train a new SAM2 model on river data
2. Use `predict_new.ipynb` to evaluate model performance and determine optimal thresholds

Output Structure:
- masks/: High-confidence binary masks (.npy files)
- masked_images/: Original images with river regions highlighted (.png files)
- low_confidence/: Masks below threshold for manual review
- processing_log.txt: Detailed processing statistics and results

## Classifier

Directories and files:
classifier
   - classifier.py: A Python file containing a RiverClassifier model class definition, with methods for initializing, training, and evaluating a rapids classification model.
   - rapids_train.py: A Python file with code to process the rapids image dataset and a driver script to train the rapids classification model. 
   - rapids_predict.py: A Python file with a driver script to perform large-scale rapids detection on river images.

The Python code for training the rapids classification models was run using the uv package manager by Astral (https://docs.astral.sh/uv/). Once uv is installed, run the following commands in the terminal to setup a Python environment with all packages required to run the model training code.

1. Change the working directory to the classifier folder in the rapid-detection-collection code archive.

2. Initialize a uv virtual environment: uv venv --python 3.12.3

3. Activate the virtual environment: 
   - Windows: source .venv/Scripts/activate
   - Mac/Linux: source .venv/bin/activate

4. Install required dependencies: uv pip install -r requirements.txt

5. Before training the rapids classifier, edit the file paths in rapids_train.py with the local machine paths for the rapids image dataset, the rapids class labels, and an output file directory. Select a model log name to identify the output files, including model weights, validation metric plots, and test dataset metrics. 

6. Then, run the training script: uv run rapids_train.py

To perform rapids detection on the unlabeled images provided in the dataset, run the following additional steps.

7. Download the flowlines_{00-04}.tar and alaska.tar files from the CIRRUS data release and place them all in a single directory.

8. Edit the file paths in rapids_predict.py with the local machine paths for the directory with the above tar files, the weights for a trained rapids classifier, and an output file directory. Select a model log name to identify the CSV containing the rapid predictions.

9. Run the prediction script using: uv run rapids_predict.py

The resulting CSV contains the predicted probability of the presence of rapids in each image.



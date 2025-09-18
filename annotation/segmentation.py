"""
River Segmentation and Rapids Annotation Interface

Author name and contact: Hannah Fluckiger, ORCID: 0009-0004-8246-1376

This script provides the main entry point for manual annotation of river images,
offering multiple annotation workflows for creating training data for river
segmentation and rapid detection models.

The script combines interactive segmentation using Segment Anything Model (SAM2)
with manual classification of river features, providing a comprehensive annotation
platform for building high-quality datasets.

Annotation Workflows:
1. Create Masks Only - Generate segmentation masks for river boundaries
2. Label Rapids Only - Classify images for presence/absence of rapids
3. Combined Workflow - Both segmentation and rapid classification
4. Label Standing Waves - Classify Undular Hydraulic Jumps (UHJs)

Setup Requirements:
Before running, create a '.user.json' file in the project root with:
{
    "user": "annotator_name",
    "metadata": "path/to/metadata.csv",
    "image_folder": "path/to/images/",
    "npy_folder": "path/to/output/masks/",
    "SAM2_CHECKPOINT_FOLDER": "path/to/sam2/checkpoints/"
}

Key Features:
- Menu-driven interface for selecting annotation tasks
- Integration with fine-tuned SAM2 model for segmentation assistance
- Automatic progress saving and resumable sessions
- Quality control prompts for annotation validation
- Comprehensive metadata tracking with timestamps

Model Integration:
- Loads fine-tuned SAM2 model for river segmentation
- Uses interactive point prompts for mask refinement
- Supports both automated and manual annotation modes
- GPU acceleration when available

Controls and Instructions:
- Mask Creation: Interactive point-based annotation with SAM2
- Rapid Classification: Keyboard-based binary classification
- UHJ Classification: Three-way classification (yes/no/maybe)
- Quality Control: Manual review and validation prompts

Output:
- Updated CSV metadata with labels and timestamps
- Binary mask files (.npy) for segmentation annotations
- Detailed annotation provenance information

Dependencies:
    - torch, numpy, json
    - SAM2 components (build_sam, sam2_image_predictor)
    - Custom modules: label, select_device
"""

import torch
import numpy as np

import json

from label import label
from select_device import select_device

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



### How to include metadata
# 
# 1. Create a new file called .user.json in the rapid-detection-collection folder
# 2. Copy everything between the triple quotes into .user.json
# 3. Change each data field to match your data accordingly
#
'''
{
    "user": "your name",
    "metadata": "path/to/csv/file",
    "image_folder": "path/to/image/folder",
    "npy_folder": "path/to/masks/folder",
    "SAM2_CHECKPOINT_FOLDER": "path/to/checkpoints"
}
'''
#
###



METADATA = ".user.json" # path to .user.json

with open('.user.json', 'r') as file:
    folders = json.load(file)

def load_model():
    device = select_device()
    model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'
    sam2_model = build_sam2(model_cfg, f'{folders["SAM2_CHECKPOINT_FOLDER"]}/sam2.1_hiera_tiny.pt', device=device)
    model = SAM2ImagePredictor(sam2_model)
    # model.model.load_state_dict(torch.load(f'{folders["SAM2_CHECKPOINT_FOLDER"]}/sam2_model_finetuned_2.pt'))
    # model.model.load_state_dict(torch.load(f'{folders["SAM2_CHECKPOINT_FOLDER"]}/sam2_model_finetuned_epoch_3.pt'))
    return model


def print_mask_instructions():
    print('Left click to add a positive point.')
    print('Right click to add a negative point.')
    print()
    print('Press "t" to label image as containing all river')
    print('Press "f" to label image as containing no river')
    print()
    print('Press "z" to remove the most recent point.')
    print('Press "q" to exit and save masks.')
    print()


def print_rapid_instuctions():
    print('Press one of the following keys to classify a given image:')
    print('\t0: Image has no rapids.')
    print('\t1: Image has rapids.')
    print()


def print_uhj_instuctions():
    print('Press one of the following keys to classify a given image:')
    print('\t0: Image has no UHJs.')
    print('\t1: Image has UHJs.')
    print()


def option_menu(option):
    option = option.strip()

    if option == '1':
        model = load_model()
        print()
        
        print_mask_instructions()

        label(folders, 'mask', model)

    elif option == '2':
        print_rapid_instuctions()

        label(folders, 'rapid')

    elif option == '3':
        model = load_model()
        print()
        
        print_rapid_instuctions() 
        print_mask_instructions()

        label(folders, 'mask_rapid', model)
    
    elif option == '4':
        print_uhj_instuctions()

        label(folders, 'uhj')

    elif option == '5':
        pass
        
    else:
        option = input('Enter a number: [1, 2, 3, 4] ')
        option_menu(option)


if __name__=='__main__':
    np.random.seed(3)

    print('1. Create masks')
    print('2. Label rapids')
    print('3. Create masks AND label rapids')
    print('4. Label standing waves')
    print('5. Quit')
    print()

    option = input('Which option do you choose? ')
    print()
    option_menu(option)

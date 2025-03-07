import re
import cv2
import json
import numpy as np
from glob import glob
from time import sleep

from label import label
from select_device import select_device
from RapidsImage import RapidsImage as Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


################################################################
# These may need to be changed 
################################################################
JSON_FOLDER = 'input'
IMAGE_FOLDER = 'input'
NPY_FOLDER = 'output'

folders = {
    'json_folder': 'input',
    'image_folder': 'input',
    'npy_folder': 'output'
}

SAM2_CHECKPOINT = 'checkpoints/sam2.1_hiera_large.pt' 
################################################################


def option_menu(option, files):
    option = option.strip()

    if option == '1':
        device = select_device()
        model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml' # don't change this line
        sam2_model = build_sam2(model_cfg, SAM2_CHECKPOINT, device=device)
        model = SAM2ImagePredictor(sam2_model)
        print()
        
        print('Left click to add a positive point.')
        print('Right click to add a negative point.')
        print('Press "z" to remove the most recent point.')
        print('Press "q" to exit and save masks.')
        print()

        label(folders, files, 'mask', model)

    elif option == '2':
        print('Press one of the following keys to classify a given image:')
        print('\t0: Image has no rapids.')
        print('\t1: Image has rapids.')
        print()

        label(folders, files, 'rapid')

    elif option == '3':
        device = select_device()
        model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml' # don't change this line
        sam2_model = build_sam2(model_cfg, SAM2_CHECKPOINT, device=device)
        model = SAM2ImagePredictor(sam2_model)
        print()
        
        print('Press one of the following keys to classify a given image:')
        print('\t0: Image has no rapids.')
        print('\t1: Image has rapids.')
        print() 

        print('Left click to add a positive point.')
        print('Right click to add a negative point.')
        print('Press "z" to remove the most recent point.')
        print('Press "q" to exit and save masks.')
        print()

        label(folders, files, 'mask_rapid', model)
    
    elif option == '4':
        print('Press one of the following keys to classify a given image:')
        print('\t0: Image has no rapids.')
        print('\t1: Image has rapids.')
        print()

        label(folders, files, 'uhj')

    elif option == '5':
        pass
        
    else:
        option = input('Enter a number: [1, 2, 3, 4] ')
        option_menu(option, files)


if __name__=='__main__':
    np.random.seed(3)

    files = glob(f'{JSON_FOLDER}/*.json')

    print('1. Create masks')
    print('2. Label rapids')
    print('3. Create masks AND label rapids')
    print('4. Label standing waves')
    print('5. Quit')
    print()

    option = input('Which option do you choose? ')
    print()
    option_menu(option, files)

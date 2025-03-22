import torch
import numpy as np
from glob import glob

from label import label
from select_device import select_device

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


################################################################
# These may need to be changed 
################################################################
JSON_FOLDER  = '' # Folder Path in Box
IMAGE_FOLDER = '' # Folder Path in Box
NPY_FOLDER   = '' # Folder Path in Box

SAM2_CHECKPOINT_FOLDER = '' # File Path in Box
################################################################


folders = {
    'json_folder' : JSON_FOLDER,
    'image_folder': IMAGE_FOLDER,
    'npy_folder'  : NPY_FOLDER
}


def load_model():
    device = select_device()
    model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'
    sam2_model = build_sam2(model_cfg, F'{SAM2_CHECKPOINT_FOLDER}/sam2.1_hiera_tiny.pt', device=device)
    model = SAM2ImagePredictor(sam2_model)
    model.model.load_state_dict(torch.load(F'{SAM2_CHECKPOINT_FOLDER}/sam2_model_finetuned_epoch_3.pt'))
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


def option_menu(option, files):
    option = option.strip()

    if option == '1':
        model = load_model()
        print()
        
        print_mask_instructions()

        label(folders, files, 'mask', model)

    elif option == '2':
        print_rapid_instuctions()

        label(folders, files, 'rapid')

    elif option == '3':
        model = load_model()
        print()
        
        print_rapid_instuctions() 
        print_mask_instructions()

        label(folders, files, 'mask_rapid', model)
    
    elif option == '4':
        print_uhj_instuctions()

        label(folders, files, 'uhj')

    elif option == '5':
        pass
        
    else:
        option = input('Enter a number: [1, 2, 3, 4] ')
        option_menu(option, files)


if __name__=='__main__':
    np.random.seed(3)

    files = glob(f'{folders["json_folder"]}/*.json')

    print('1. Create masks')
    print('2. Label rapids')
    print('3. Create masks AND label rapids')
    print('4. Label standing waves')
    print('5. Quit')
    print()

    option = input('Which option do you choose? ')
    print()
    option_menu(option, files)

import re
import cv2
import json
import numpy as np
from RapidsImage import RapidsImage as Image


def is_valid_json(file, label_type):
    
    if label_type == 'mask':
        return file['map'] == ''
    
    elif label_type == 'rapid':
        if 'class' in file:
            file['rapid_class'] = file.pop('class')
        if 'rapids_class' in file:
            file['rapid_class'] = file.pop('rapids_class')
        return file['rapid_class'] == ''
    
    elif label_type == 'mask_rapid':
        if 'class' in file:
            file['rapid_class'] = file.pop('class')
        return file['map'] == '' and file['rapid_class'] == ''
    
    elif label_type == 'uhj':
        if 'uhj_class' not in file:
            file['uhj_class'] = ''
        return (file['uhj_class'] == '') and (file['rapid_class'] == 1)
        

def display_image(my_image, label_type):
    
    cv2.namedWindow(winname='image') 
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image', my_image.click_event) 

    if label_type == 'mask':
        while True:
            cv2.imshow('image', my_image.image_mask)
            key = cv2.waitKey(1) 
            
            if key == ord('t'):
                my_image.remove_last()
                break
            elif key == ord('f'):
                my_image.remove_last()
                break
            
            elif key == ord('z'):
                my_image.remove_last()
            elif key == ord('q'):
                break
    
    elif label_type in ['rapid', 'uhj']:
        while True:
            cv2.imshow('image', my_image.image_mask)
            key = cv2.waitKey(1) 
            if key == ord('0') or key == ord('1'):
                my_image.rapid_class = int(key - ord('0'))
                break
    
    elif label_type == 'mask_rapid':
        while True:
            cv2.imshow('image', my_image.image_mask)
            key = cv2.waitKey(1) 

            if key == ord('0') or key == ord('1'):
                my_image.rapid_class = int(key - ord('0'))

                my_image.set_textmsg(f'Image has been classified as having {"no " if my_image.rapid_class == 0 else ""}rapids.')
                my_image.display_image()

            if key == ord('z'):
                my_image.remove_last()
            
            elif key == ord('q'):
                break
    
    cv2.destroyAllWindows()
    return my_image
    

def label(folders, files, label_type, model=None):
    
    for file_name in files:
        with open(file_name, 'r') as f:
            file = json.load(f)

        if not is_valid_json(file, label_type):
            continue

        
        file['labeled_by_human'] = 1

        signature = re.split(r'[/\\]', file['image'])[-1].rsplit('.', 1)[0]

        print(f'Image: {file['name']}')
        print()

        if label_type in ['rapid', 'mask_rapid']:
            msg = 'Does this image contain rapids? [0/1]'
        elif label_type == 'uhj':
            msg = 'Does this image contain UHJs? [0/1]'
        else:
            msg = ''

        try:
            my_image = Image(
                image=cv2.imread(f'{folders["image_folder"]}/{signature}.png', 1),
                predictor=model,
                has_textbox=(label_type!='mask'),
                msg=msg
            )
        except Exception:
            print()
            continue

        my_image = display_image(my_image, label_type)

        if my_image.rapid_class >= 0:
            if label_type in ['rapid', 'mask_rapid']:
                print(f'Image has been classified as having {"no " if my_image.rapid_class == 0 else ""}rapids.')
                file['rapid_class'] = my_image.rapid_class
                if my_image.rapids_class == 0:
                    file['uhj_class'] = 0
            elif label_type == 'uhj':
                print(f'Image has been classified as having {"no " if my_image.rapid_class == 0 else ""}UHJs.')
                file['uhj_class'] = my_image.rapid_class

        if label_type in ['mask', 'mask_rapid']:
            save_npy = input('Would you like to save your masks for this image? [y/n] ')
            if save_npy.lower() == 'y' or save_npy.lower() == 'yes':
                npy_file_name = f'{signature}.npy'
                np.save(
                    f'{folders["npy_folder"]}/{npy_file_name}',
                    my_image.masks, 
                )
                file['map'] = npy_file_name
            else:
                file['map'] = None
        
        with open(f'{folders["json_folder"]}/{signature}.json', 'w') as f:
            json.dump(file, f, indent=4)

        again = input('Would you like to continue? [y/n] ') if len(files) > 0 else 'n'
        if again.lower() == 'n' or again.lower() == 'no':
            return
        print()

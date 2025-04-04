import cv2
import time
import numpy as np
import pandas as pd
from RapidsImage import RapidsImage as Image

def is_valid_json(line, label_type):
    
    if label_type == 'mask':
        return np.isnan(float(line['map']))
    
    elif label_type == 'rapid':
        return np.isnan(float(line['rapid_class']))
    
    elif label_type == 'mask_rapid':
        return np.isnan(float(line['map'])) and np.isnan(float(line['rapid_class']))
    
    elif label_type == 'uhj':
        return np.isnan(float(line['uhj_class'])) and (float(line['rapid_class']) == 1)
        

def display_image(my_image, label_type):
    
    cv2.namedWindow(winname='image') 
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image', my_image.click_event) 

    if label_type == 'mask':
        while True:
            cv2.imshow('image', my_image.image_mask)
            key = cv2.waitKey(1) 
            
            if key == ord('t'):
                my_image.set_masks(np.ones(my_image.masks.shape))
            elif key == ord('f'):
                my_image.set_masks(np.zeros(my_image.masks.shape))
            
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

            if key == ord('t'):
                my_image.set_masks(np.ones(my_image.masks.shape))
            elif key == ord('f'):
                my_image.set_masks(np.zeros(my_image.masks.shape))

            if key == ord('z'):
                my_image.remove_last()            
            elif key == ord('q'):
                break
    
    cv2.destroyAllWindows()
    return my_image
    

def label(folders, label_type, model=None):
    
    # TODO lat/lon should be strs
    df = pd.read_csv(folders["metadata"])
    
    for i in range(len(df)):
        
        line = df.loc[i][:]
        if not is_valid_json(line, label_type):
            continue
        
        line['labeled_by_human'] = 1

        df.loc[i] = line

        print(f'Image: {line["name"]}')
        print()

        if label_type in ['rapid', 'mask_rapid']:
            msg = 'Does this image contain rapids? [0/1]'
        elif label_type == 'uhj':
            msg = 'Does this image contain UHJs? [0/1]'
        else:
            msg = ''

        try:
            my_image = Image(
                image=cv2.imread(f'{folders["image_folder"]}/{line["image"]}.png', 1),
                predictor=model,
                has_textbox=(label_type!='mask'),
                msg=msg
            )
        except Exception:
            print()
            continue

        my_image = display_image(my_image, label_type)

        today = np.floor(time.time()) # this is platform dependent? may want to find a different solutiont 

        if my_image.rapid_class >= 0:
            if label_type in ['rapid', 'mask_rapid']:
                print(f'Image has been classified as having {"no " if my_image.rapid_class == 0 else ""}rapids.')
                line['rapid_labeled_on'] = today
                line['rapid_class'] = my_image.rapid_class

                if my_image.rapid_class == 0:
                    line['uhj_labeled_on'] = today
                    line['uhj_class'] = 0
            
            elif label_type == 'uhj':
                print(f'Image has been classified as having {"no " if my_image.rapid_class == 0 else ""}UHJs.')
                line['uhj_labeled_on'] = today
                line['uhj_class'] = my_image.rapid_class

        if label_type in ['mask', 'mask_rapid']:
            line['mask_created_on'] = today
            save_npy = input('Would you like to save your masks for this image? [y/n] ')
            if save_npy.lower() == 'y' or save_npy.lower() == 'yes':
                np.save(
                    f'{folders["npy_folder"]}/{f'{line["image"]}.npy'}',
                    my_image.masks, 
                )
                line['map'] = 1
            else:
                line['map'] = 0
        
        df.loc[i] = line
        df.to_csv(folders["metadata"], index=False)

        again = input('Would you like to continue? [y/n] ') if i < len(df) - 1 else 'n'
        if again.lower() == 'n' or again.lower() == 'no':
            return
        print()

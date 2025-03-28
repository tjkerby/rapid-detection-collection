import re
import json
from glob import glob

import firebase_admin
from firebase_admin import credentials, firestore


FIELDS = [
    "name", "longitude", "latitude", "zoom", "image", "map", 
    "rapid_class", "uhj_class", "labeled_by_human", 
    "mask_created_on", "rapid_labeled_on", "uhj_labeled_on"
]


i = 0
if __name__ == '__main__':

    with open(".private_key", 'r') as f:
        key = f.readline()
    
    cred = credentials.Certificate(key)
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    doc_ref = db.collection("images")

    folder = '../json_folder'
    files = glob(f'{folder}/*.json')

    for file_name in files:

        with open(file_name, 'r') as f:
            file = json.load(f)
        
        if 'in_database' in file:
            continue
        
        if 'image' not in file:
            file['image'] = ''
        else:
            file['image'] = re.split(r'[/\\]', file['image'])[-1].rsplit('.', 1)[0]

        if 'name' not in file:
            file['name'] = ''

        if 'longitude' not in file:
            file['longitude'] = ''

        if 'latitude' not in file:
            file['latitude'] = ''

        if 'zoom' not in file:
            file['zoom'] = ''

        if ('map' not in file) or (file['map'] == ''):
            file['map'] = ''
        elif (file['map'] == None) or (file['map'] == 0):
            file['map'] = 0
        else:
            file['map'] = 1

        if 'class' in file:
            file['rapid_class'] = file.pop('class')
        if 'rapids_class' in file:
            file['rapid_class'] = file.pop('rapids_class')
        if 'rapid_class' not in file:
            file['rapid_class'] = ''

        if 'uhj_class' not in file:
            file['uhj_class'] = ''

        if 'labeled_by_human' not in file:
            file['labeled_by_human'] = ''

        if 'mask_created_on' not in file:
            file['mask_created_on'] = ''

        if 'rapid_labeled_on' not in file:
            file['rapid_labeled_on'] = ''

        if 'uhj_labeled_on' not in file:
            file['uhj_labeled_on'] = ''

        for field in file:
            if field not in FIELDS:
                print(f'{file_name}\n{field}: {file[field]}\n')

        doc_ref.document(file['image']).set(file)

        file['in_database'] = True

        with open(file_name, 'w') as f:
            json.dump(file, f, indent=4)

        if i >= 1_000:
            break
        i += 1
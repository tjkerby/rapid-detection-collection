import pandas as pd
import numpy as np
from PIL import Image

from helper_functions import show_masks

dat = pd.read_csv('segmentation_results.csv')
dat['score_new'] = dat['score_new'].str.strip('[]')
dat['score_new'] = dat['score_new'].astype('float32')

dat = dat.sort_values(by='score_new', ascending=False)

rows, cols = dat.shape
for i in range(rows):
    img = Image.open(dat['img'].iloc[i])
    if img.mode != 'RGB':
        img = img.convert('RGB')

    mask = np.load(dat['mask_new'].iloc[i])
    score = [dat['score_new'].iloc[i]]

    show_masks(img, mask, score)

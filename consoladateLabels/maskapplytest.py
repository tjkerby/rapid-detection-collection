import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rootPath = './'

mask_dir = rootPath + 'RapidsMask'
data = [f.replace('.npy', '') for f in os.listdir(mask_dir) if f.endswith(('.npy'))]

for i in range(len(data)):
    imgPath = rootPath + 'rapids/' + data[i] + '.png'
    maskPath = rootPath + 'RapidsMask/' + data[i] + '.npy'

    image = Image.open(imgPath)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = np.array(image)
    mask = np.load(maskPath)
    mask = np.squeeze(mask)
    # Expand mask to 3 channels and multiply with image
    mask_expanded = mask[:, :, np.newaxis]
    masked_array = image_array * mask_expanded
    masked_image = masked_array.astype(np.uint8)

    # Save masked image
    masked_image = Image.fromarray(masked_image)
    masked_image.save(rootPath + 'Rapidsmasked_images/' + data[i] + '_masked.png')
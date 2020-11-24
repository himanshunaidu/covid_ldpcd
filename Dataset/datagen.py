#ImageDataGenerator is used to create more images from the existing COVID covid dataset images
import os
import shutil
#import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image, ImageOps

from config import covid_std_path, gen_path

if __name__ == '__main__':
    datagen = ImageDataGenerator(rotation_range=30,  width_shift_range=0.2,  height_shift_range=0.2, shear_range=0.3, zoom_range=0.2, horizontal_flip=True, fill_mode='constant')
    data_files = os.listdir(covid_std_path)
    data = np.zeros(shape=(len(data_files), 227, 227, 1))
    data_index = 0

    for filename in data_files:
        image_path = os.path.join(covid_std_path, filename)
        img = Image.open(image_path)
        #print(img_arr.shape)
        if img.mode != 'L':
            img = ImageOps.grayscale(img)
        img = img.resize((227, 227))
        img_arr = np.asarray(img)
        #2D to 3D
        img_arr = np.reshape(img_arr, newshape=img_arr.shape+(1,))
        
        data[data_index] = img_arr
        data_index = data_index+1

    print(data.shape)

    #import random
    #for i in range(10):
    #    print(data[random.randint(0, data.shape[0])])

    count = data.shape[0]
    i = 0
    for batch in datagen.flow(data, batch_size=150, save_to_dir=gen_path, save_prefix='covid', save_format='png'):
        i = i+1
        print(f"Batch {i} Done")
        if (i>=10):
            break

    print('Done')
#Extensions are checked to assess the composition of the images in the datasets
import os
import shutil
#import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from config import data_path

if __name__ == '__main__':
    data_files = os.listdir(data_path)
    jpg_checks, jpeg_checks, png_checks = [], [], []
    #print(data_files)

    for i in data_files:
        filename, file_ext = os.path.splitext(i)
        if file_ext=='.jpg':
            jpg_checks.append(i)
        elif file_ext=='.jpeg':
            jpeg_checks.append(i)
        elif file_ext=='.png':
            png_checks.append(i)
        else:
            print(file_ext)

    print(len(jpg_checks), len(jpeg_checks), len(png_checks))

    #Dimension check
    min_height, min_width = 10000, 10000
    avg_height, avg_width = 0, 0
    count = 0

    for filename in data_files:
        image_path = os.path.join(data_path, filename)
        img = Image.open(image_path)
        #print(img_arr.shape)
        if img.mode != 'L':
            img = ImageOps.grayscale(img)
        img_arr = np.asarray(img)
        (height, width) = img_arr.shape
        if height<300 or width<300:
            print(height, width)
        if min_height>height:
            min_height = height
        if min_width>width:
            min_width = width
        
        avg_height = avg_height+height
        avg_width = avg_width+width
        count = count+1

    print(min_height, min_width)
    print(avg_height/count, avg_width/count)
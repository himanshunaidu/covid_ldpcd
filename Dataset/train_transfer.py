#Move chunks of dataset as per requirement
import os
import shutil
#import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image, ImageOps

from config import train_path, test_path, normal_path_name, normal_std_path_name, normal_ldp_path_name
from config import bacterial_path_name, bacterial_std_path_name, bacterial_ldp_path_name
from config import viral_path_name, viral_std_path_name, viral_ldp_path_name

def normal_transfer(filenames, read_path, write_path):
    count = 0
    for filename in filenames:
        src_path = os.path.join(read_path, filename)
        dst_path = os.path.join(write_path, 'covid_'+filename)
        #img = Image.open(image_path)
        shutil.copyfile(src=src_path, dst=dst_path)
        count = count+1
    return count

def std_transfer(filenames, read_path, write_path):
    count = 0
    for filename in filenames:
        src_path = os.path.join(read_path, filename)
        #Destination image must be stored in png
        new_filename, file_extension = os.path.splitext(filename)
        new_filename = new_filename+'.png'
        dst_path = os.path.join(write_path, new_filename)

        img = Image.open(src_path)
        if img.mode != 'L':
            img = ImageOps.grayscale(img)
        img = img.resize((227, 227))
        img.save(dst_path, "PNG")
        count = count+1
        print(dst_path)
    return count

if __name__ == '__main__':

    #Normal to Std Directory
    #normal_files = os.listdir(os.path.join(test_path, normal_path_name))
    #result = std_transfer(normal_files, os.path.join(test_path, normal_path_name), os.path.join(test_path, normal_std_path_name))
    #print(result)

    #Bacterial to Std Directory
    #bacterial_files = os.listdir(os.path.join(train_path, bacterial_path_name))
    #result = std_transfer(bacterial_files, os.path.join(train_path, bacterial_path_name), os.path.join(train_path, bacterial_std_path_name))
    #print(result)

    #Viral to Std Directory
    #viral_files = os.listdir(os.path.join(train_path, viral_path_name))
    #result = std_transfer(viral_files, os.path.join(train_path, viral_path_name), os.path.join(train_path, viral_std_path_name))
    #print(result)
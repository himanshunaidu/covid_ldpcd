#Split the datasets into the required train and test sets
import os
import shutil
#import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

from config import train_path, test_path, covid_std_path_name

main_read_path = train_path
main_write_path = train_path

def splitFileData(arr_x, arr_y):
    x_train, x_test, y_train, y_test = train_test_split(arr_x, arr_y, test_size=0.43, random_state=42)
    print(len(x_train))
    print(len(x_test))
    return x_train, x_test

def writeFileData(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

def moveFiles(filenames, src_path, dst_path):
    index = 0
    file = ''
    for filename in filenames:
        source = os.path.join(src_path, filename)
        shutil.move(source, dst_path)
        file = source
        index = index+1
        #if index>0:
        #    break
    print(index)
    return file

if __name__ == '__main__':
    #Get all the COVID data in train directory and split
    #data_files = os.listdir(data_path)
    #data_y = np.zeros(len(data_files))
    #data_file_train, data_file_test = splitFileData(data_files, data_y)

    #Write the split data to separate text files
    #writeFileData(file_train_path, data_file_train)
    #writeFileData(file_test_path, data_file_test)

    #Get all COVID data and split
    data_files = os.listdir(os.path.join(train_path, covid_std_path_name))
    data_y = np.zeros(len(data_files))
    data_file_train, data_file_test = splitFileData(data_files, data_y)

    moveFiles(data_file_test, os.path.join(train_path, covid_std_path_name), os.path.join(test_path, covid_std_path_name))

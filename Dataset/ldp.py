#LDP is performed through custom code
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import time
#For GPU
from numba import vectorize, guvectorize, jit, cuda 

from config import test_path, covid_path_name, covid_std_path_name, covid_ldp_path_name

east= [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]
north_east= [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]
north= [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]
north_west= [[5, 5, -3], [5, 0, -3], [-3, -3, -3]]
west= [[5, -3, -3], [5, 0, -3], [5, -3, -3]]
south_west= [[-3, -3, -3], [5, 0, -3], [5, 5, -3]]
south= [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]
south_east= [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]

directions_dict = {'east': east, 'north-east': north_east, 'north': north, 'north-west': north_west, 'west': west, 'south-west': south_west, 'south': south, 'south-east': south_east}
directions = ['east', 'north-east', 'north', 'north-west', 'west', 'south-west', 'south', 'south-east']
paths = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)] #Cycle goes from south (2, 1) to south-west (2, 0) anti-clockwise
k = 3 #Top k absolute values in the ldp matrix would be considered

#show neighborhood for a pixel
def show_nh(img, x, y):
    nh = np.zeros((3, 3))
    nh[1][1] = img[x][y]
    (center_x, center_y) = (1, 1)
    for (i, j) in paths:
        try:
            nh[center_x+i][center_y+j] = img[x+i][y+j]
            #print(nh[center_x+i][center_y+j], i, j)
        except:
            pass
    return nh

#Get the value of the pixel associated with a direction
#@guvectorize(['int64(int64[:, :, :], int64, int64, int64)'], '(x, y, z),(),(),()->()', target ="cuda")
def get_pixel(img, x, y, direction):
    new_value = 0
    mask = directions_dict[direction]
    (mask_cx, mask_cy) = (1, 1) #mask center co-ordinates
    for (i, j) in paths:
        try:
            new_value = new_value + mask[mask_cx+i][mask_cy+j]*img[x+i][y+j]
            #print(new_value, i, j, img[x+i][y+j], mask[mask_cx+i][mask_cy+j])
        except:
            pass
    #print(new_value)
    return new_value

#@guvectorize(['int64(int64[:, :, :], int64, int64)'], '(x, y, z),(),()->()', target ="cuda")
def ldp_calculate_pixel(img, x, y):
    center = img[x][y] 
    val_ar, sort_val_ar = [], []

    #Get pixel value for each direction
    for direction in directions:
        val_ar.append(get_pixel(img, x, y, direction))
        sort_val_ar.append(abs(get_pixel(img, x, y, direction)))
    
    #Find the k largest absolute pixel values and give them value 1
    krrish = np.zeros((8))
    sort_val_ar.sort(reverse=True)
    k_val = [sort_val_ar[0], sort_val_ar[1], sort_val_ar[2]]

    for i in range(len(k_val)):
        for j in range(len(val_ar)):
            #print(val_ar[j], k_val[i])
            if abs(val_ar[j])==abs(k_val[i]) and krrish[j]!=1:
                krrish[j] = 1
                break
        #print(krrish)
    #print(krrish)
    #print(val_ar)

    #Do the vector dot product with the power values and get the final pixel value and assign that to center pixel
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    center_val = 0
      
    for i in range(len(val_ar)): 
        center_val += krrish[i] * power_val[i]
    
    return center_val

#@guvectorize(['int64(int64[:, :, :], int64, int64)'], '(x, y, z),(),()->()', target ="cuda")
def ldp_calculate(img, filename, dst_path):
    height, width = img.shape
    img_ldp = np.zeros((height, width))
    for i in range(1, height-1):
        for j in range(1, width-1):
            img_ldp[i, j] = ldp_calculate_pixel(img, i, j)
    img_ldp = Image.fromarray(img_ldp)
    img_ldp.convert('L').save(os.path.join(dst_path, filename))
    return img_ldp

if __name__ == '__main__':
    covid_std_files = os.listdir(os.path.join(test_path, covid_std_path_name))
    covid_std_data = np.zeros(shape=(len(covid_std_files), 227, 227))
    index = 0

    #Get all the images and insert into std_data for ldp calculation

    for filename in covid_std_files:
        image_path = os.path.join(os.path.join(test_path, covid_std_path_name), filename)
        #Image is already 227*227 and grayscale, no need of any extra operations
        img = Image.open(image_path)
        img_arr = np.asarray(img)
        
        covid_std_data[index] = img_arr
        index = index+1

    print(covid_std_data.shape)

    #std_files and std_data have the same index sequence
    index = 0
    start_time = time.time()
    for index in range(len(covid_std_data)):
        ldp = ldp_calculate(covid_std_data[index], covid_std_files[index], os.path.join(test_path, covid_ldp_path_name))
        print(f"Time required {time.time() - start_time}")
        print(ldp, index)
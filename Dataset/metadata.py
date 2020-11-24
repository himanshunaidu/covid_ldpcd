#The metadata of the Cohen JP COVID dataset is checked to see the characteristics of the dataset
import os
import shutil
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import dir_path, data_path
os.chdir(dir_path)

def get_image(df):
    folder = df['folder']
    filename = df['filename']
    image_path = os.path.join(folder, filename)
    img = Image.open(image_path)
    img.show()

def copy_image(df):
    folder = df['folder']
    filename = df['filename']
    src_path = os.path.join(folder, filename)
    dst_path = os.path.join(data_path, filename)
    shutil.copyfile(src=src_path, dst=dst_path)
    print(src_path)

def traverse_df(df):
    for i in range(len(df)):
        #get_image(df.iloc[i])
        copy_image(df.iloc[i])

if __name__ == '__main__':
    metadata = pd.read_csv('metadata.csv')
    print(len(metadata))

    covid_chest_pa = metadata.loc[(metadata['folder'] == 'images') & (metadata['view'] == 'PA') & (metadata['finding'] == 'COVID-19')]
    covid_chest_ap = metadata.loc[(metadata['folder'] == 'images') & (metadata['view'] == 'AP') & (metadata['finding'] == 'COVID-19')]
    covid_chest_ap_supine = metadata.loc[(metadata['folder'] == 'images') & (metadata['view'] == 'AP Supine') & (metadata['finding'] == 'COVID-19')]
    covid_chest_axial = metadata.loc[(metadata['folder'] == 'images') & (metadata['view'] == 'Axial') & (metadata['finding'] == 'COVID-19')]
    covid_chest_coronal = metadata.loc[(metadata['folder'] == 'images') & (metadata['view'] == 'Coronal') & (metadata['finding'] == 'COVID-19')]
    covid_chest_l = metadata.loc[(metadata['folder'] == 'images') & (metadata['view'] == 'L') & (metadata['finding'] == 'COVID-19')]

    #Mostly use only PA, AP and AP Supine
    #print(len(covid_chest_ap), len(covid_chest_ap_supine), len(covid_chest_axial), len(covid_chest_coronal), len(covid_chest_l), len(covid_chest_pa))
    print(len(covid_chest_ap)+len(covid_chest_ap_supine)+len(covid_chest_pa))

    #Execute the following to copy the AP, AP_Supine and PA images to the training directory
    #traverse_df(covid_chest_ap)
    #traverse_df(covid_chest_ap_supine)
    #traverse_df(covid_chest_pa)
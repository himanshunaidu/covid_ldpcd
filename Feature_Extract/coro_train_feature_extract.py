#Used to extract all the image features generated on the COVID train dataset by Dark11 Network
#and insert in MySQL database into table train_image_features4
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import math
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, LeakyReLU, Flatten, Dense, BatchNormalization
from tensorflow.keras.applications import xception
from tensorflow.keras.models import Model
import mysql.connector

from dataset import genDataset3DWithNames
from pretrain_coro_1 import createNetwork
from config import covid_train_path, normal_train_path, bacterial_train_path, viral_train_path
from config import network_best_load_path
from config import mydb, coronet_train_table

print(mydb)
mycursor = mydb.cursor()

length = 10

def printSumFeatures(features, outputs):

    for index in range(len(features)):
        sum, count = 0, 0
        for i in features[index]:
            if abs(i)>=1.0:
                count = count+1
            sum = sum + i
        print(sum, count/len(features[index])*100, outputs[index])

def insertIntoDB(db, cursor, image, dense, dense1, outputs):
    sql = "INSERT INTO " + coronet_train_table + "(image_name, dense, dense_1, class) VALUES (%s, %s, %s, %s)"

    for index in range(len(dense)):
        dense_arg = ' '.join(str(i) for i in dense[index])
        dense_arg1 = ' '.join(str(i) for i in dense1[index])
        val = (image[index], dense_arg, dense_arg1, float(outputs[index]))
        #print(val[3])
        try:
            cursor.execute(sql, val)
            print('Inserted ', cursor.lastrowid)
            if cursor.lastrowid==0:
                raise Exception('Not enough')
        except:
            print('Exception Occured', val)
            db.commit()
            return
    db.commit()


x_train, y_train, path_train = genDataset3DWithNames(normal_train_path, bacterial_train_path, viral_train_path, covid_train_path)
x_train = xception.preprocess_input(x_train)
#x_train = np.expand_dims(x_train, -1)
print(x_train.shape, y_train.shape)

model = createNetwork()
print(model.summary())
model.load_weights(network_best_load_path)

model_dense = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('predictions').output)
#print(model_dense.get_weights())
model_dense_1 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('dense').output)
#print(model_dense_1.get_weights())

#model_dense_2 = model.get_layer('avg_pool')

features = model_dense.predict(x_train) #[0:length, :])
print('Features', len(features))
#printSumFeatures(features, y_test) #y_train#[0:length])
features1 = model_dense_1.predict(x_train) #[0:length, :])
print('Features 1', len(features1))
#printSumFeatures(features1, y_test) #y_train#[0:length])

if __name__ == '__main__':
    insertIntoDB(mydb, mycursor, path_train, features, features1, y_train)
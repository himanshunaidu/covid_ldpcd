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
from pretrain_coro_ldp1 import createNetwork
from config import covid_ldp_test_path, normal_ldp_test_path, bacterial_ldp_test_path, viral_ldp_test_path
from config import network_ldp_best_load_path
from config import mydb, coronet_test_table

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

def updateDB(db, cursor, image, dense, dense1, outputs):
    #You can use sequential index for id, since the dataset is randomized in predictable manner
    sql = "UPDATE " + coronet_test_table + " set image_ldp_path=%s, dense_ldp=%s, dense_ldp_1=%s, ldp_class=%s where id=%s"
    id_index = 0
    print(sql)

    for index in range(len(dense)):
        id_index = id_index+1
        dense_ldp_arg = ' '.join(str(i) for i in dense[index])
        dense_ldp_arg1 = ' '.join(str(i) for i in dense1[index])
        val = (image[index], dense_ldp_arg, dense_ldp_arg1, float(outputs[index]), id_index)
        #print(val[3])
        #try:
        cursor.execute(sql, val)
        print('Updated ', cursor.lastrowid)
            #if cursor.lastrowid<0:
                #raise Exception('Not enough')
    db.commit()

x_train, y_train, path_train = genDataset3DWithNames(normal_ldp_test_path, bacterial_ldp_test_path, viral_ldp_test_path, covid_ldp_test_path)
x_train = xception.preprocess_input(x_train)
#x_train = np.expand_dims(x_train, -1)
print(x_train.shape, y_train.shape)

model = createNetwork()
print(model.summary())
model.load_weights(network_ldp_best_load_path)

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
    updateDB(mydb, mycursor, path_train, features, features1, y_train)
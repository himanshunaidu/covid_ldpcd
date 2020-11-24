#Network that will train and test on the normal COVID dataset images
import os
import tensorflow as tf
import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, LeakyReLU, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet50
import tensorflow.keras.backend as kb

from dataset import genDataset, genDataset3D, genDatasetFiles, image_height, image_width, image_mode
from stats import plot_confusion_matrix
from config import covid_train_path, normal_train_path, bacterial_train_path, viral_train_path 
from config import covid_test_path, normal_test_path, bacterial_test_path, viral_test_path
from config import network_save_path, network_load_path, network_best_load_path, network_best_save_path

alpha = 0.15

#For padding
def plus_one_pad(tensor, mode):
    return tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], mode)

#For initialization
def initializer(length):
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0) #1/math.sqrt(length/2))

#Network
def createNetwork():
    model = resnet50.ResNet50(weights='imagenet')
    x = Dense(4, activation='softmax')(model.get_layer('predictions').output)
    model = Model(model.input, x)
    return model

def custom_loss(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = scce(y_true, y_pred)

    for index in range(len(y_true)):
        if y_true[index][0]==3:
            loss_1 = abs(1-y_pred[index][3])
            loss = loss + loss_1
    return loss

if __name__ == '__main__':
    #The arguments list for genDataset are being deliberately flipped so that Covid has class 2 and Normal has class 0
    #x_train, y_train = genDataset3D(normal_train_path, bacterial_train_path, viral_train_path, covid_train_path)
    #x_train = np.expand_dims(x_train, -1)
    #x_train = resnet50.preprocess_input(x_train)
    #print(x_train.shape, y_train.shape)
    x_test, y_test = genDataset3D(normal_test_path, bacterial_test_path, viral_test_path, covid_test_path)
    #x_test = np.expand_dims(x_test, -1)
    x_test = resnet50.preprocess_input(x_test)
    print(x_test.shape, y_test.shape)

    model = createNetwork()
    model.compile(optimizer='adam',
                loss=custom_loss,
                metrics=['accuracy'])

    print(model.summary())

    model.load_weights(network_best_load_path)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=network_save_path, save_weights_only=True, verbose=1)
    #cp_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=network_best_save_path, save_weights_only=True, monitor='val_accuracy', save_best_only=True, verbose=1)
    #r = model.fit(x_train, y_train, batch_size=10, validation_data=(x_test, y_test), epochs=25, callbacks=[cp_callback, cp_best_callback])

    # Plot confusion matrix
    p_test = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))

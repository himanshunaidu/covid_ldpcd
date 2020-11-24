#Network that will train and test on the normal COVID dataset images
import tensorflow as tf
import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, LeakyReLU, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model

from dataset import genDataset, genDatasetFiles, image_height, image_width, image_mode
from stats import plot_confusion_matrix
from config import covid_train_path, normal_train_path, bacterial_train_path, viral_train_path
from config import covid_test_path, normal_test_path, bacterial_test_path, viral_test_path
from config import network_save_path, network_load_path

alpha = 0.15

#For padding
def plus_one_pad(tensor, mode):
    return tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], mode)

#For initialization
def initializer(length):
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0) #1/math.sqrt(length/2))

#Network
def createNetwork():
    i = Input(shape=(image_height, image_width, image_mode))
    x = Conv2D(8, (3, 3), padding='valid', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(i)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(16, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = plus_one_pad(x, "SYMMETRIC")
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(16, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = plus_one_pad(x, "SYMMETRIC")
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(32, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)

    #Filler pad layers for continuity (We are using 227x227 instead of 256x256)
    #x = plus_one_pad(x, "SYMMETRIC")
    #x = plus_one_pad(x, "SYMMETRIC")
    x = MaxPool2D((2, 2), strides=2)(x)
    #x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = plus_one_pad(x, "SYMMETRIC")
    x = plus_one_pad(x, "SYMMETRIC")
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(64, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2)(x)

    x = Conv2D(256, (1, 1), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = plus_one_pad(x, "SYMMETRIC")
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = plus_one_pad(x, "SYMMETRIC")
    x = BatchNormalization()(x) #Dark 4
    #x = MaxPool2D((2, 2), strides=2)(x)
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    #x = BatchNormalization()(x) #Dark 4
    x = Conv2D(2, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer(3*3))(x)
    x = BatchNormalization()(x)
    #x = MaxPool2D((2, 2), strides=2)(x)
    #print(x)

    x = Flatten()(x)
    x = Dense(2304, activation=LeakyReLU(alpha=0.3))(x) #2304 = 12x12x64/4
    x = Dropout(0.3)(x)
    x = Dense(2304, activation=LeakyReLU(alpha=0.3))(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation=LeakyReLU(alpha=0.3))(x)

    x = Dense(4, activation='softmax')(x)

    #print(d1, d2, d3)
    model = Model(i, x)
    return model

if __name__ == '__main__':

    x_train, y_train = genDataset(covid_train_path, normal_train_path, bacterial_train_path, viral_train_path)
    x_train = np.expand_dims(x_train, -1)
    print(x_train.shape, y_train.shape)
    x_test, y_test = genDataset(covid_test_path, normal_test_path, bacterial_test_path, viral_test_path)
    x_test = np.expand_dims(x_test, -1)
    print(x_test.shape, y_test.shape)

    model = createNetwork()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    print(model.summary())

    model.load_weights(network_load_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=network_save_path, save_weights_only=True, verbose=1)
    r = model.fit(x_train, y_train, batch_size=10, validation_data=(x_test, y_test), epochs=10, callbacks=[cp_callback])

    # Plot loss per iteration
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    print(r.history['loss'], r.history['val_loss'])

    plt.show()

    # Plot accuracy per iteration
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    print(r.history['accuracy'], r.history['val_accuracy'])

    plt.show()

    # Plot confusion matrix
    p_test = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))

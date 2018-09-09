import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, Sequential
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# ignore warning messages
warnings.filterwarnings('ignore')

X_train = pickle.load(open('X_train_gray.pickle','rb'))/255
y_train = pickle.load(open('y_train_gray.pickle','rb'))/255
# X_test = pickle.load(open('X_test.pickle','rb'))

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def fit_model(x_train, y_train):
    with tf.device('/gpu:0'):

        model = Sequential()
        model.add(Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same', input_shape=(128,128,1)))
        model.add(Dropout(0.1))
        model.add(Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(256,(3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv2D(256,(3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(1024, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model.add(Dropout(0.1))
        model.add(Conv2D(1024, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))

        model2 = Sequential()
        model2.add(Conv2DTranspose(512, (2,2), strides=(2,2), padding='same'))

        merged1 = concatenate([model, model2], axis=1)

        model3 = Sequential()
        model3.add(Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model3.add(Dropout(0.1))
        model3.add(Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model3.add(Conv2DTranspose(256, (2,2), strides=(2,2), padding='same'))

        merged2 = concatenate([merged1, model3], axis=1)

        model4 = Sequential()
        model4.add(Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model4.add(Dropout(0.1))
        model4.add(Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model4.add(Conv2DTranspose(128, (2,2), strides=(2,2), padding='same'))

        merged3 = concatenate([merged2, model4], axis=1)

        model5 = Sequential()
        model5.add(Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model5.add(Dropout(0.1))
        model5.add(Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model5.add(Conv2DTranspose(64, (2,2), strides=(2,2), padding='same'))

        merge4 = concatenate([merged3, model5])

        model6 = Sequential()
        model6.add(Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same'))
        model6.add(Conv2D(1, (1,1), activation='sigmoid'))

        result = concatenate([merge4, model6])
        result.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

        checkpoint_model = ModelCheckpoint('my_unet_gray.h5', verbose=1, save_best_only=True)

        results = result.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=10, callbacks=[checkpoint_model])

    return results


if __name__ == '__main__':
    fit_model(X_train, y_train)

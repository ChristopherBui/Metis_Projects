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

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
from tensorflow import metrics

# ignore warning messages
warnings.filterwarnings('ignore')

X_train = pickle.load(open('X_train.pickle','rb'))/255
y_train = pickle.load(open('y_train.pickle','rb'))
# X_test = pickle.load(open('X_test.pickle','rb'))

def mean_iou_(y_true, y_pred):
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

        inputs = Input((128,128,3))

        conv1 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        drop1 = Dropout(0.1)(conv1)
        conv1 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop1)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer='he_normal', padding='same')(pool1)
        drop2 = Dropout(0.1)(conv2)
        conv2 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer='he_normal', padding='same')(drop2)
        pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

        conv3 = Conv2D(256,(3,3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
        drop3 = Dropout(0.1)(conv3)
        conv3 = Conv2D(256,(3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop3)
        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

        conv4 = Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
        drop4 = Dropout(0.1)(conv4)
        conv4 = Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop4)
        pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

        conv5 = Conv2D(1024, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(pool4)
        drop5 = Dropout(0.1)(conv5)
        conv5 = Conv2D(1024, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop5)

        up6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv5)
        up6 = concatenate([up6, conv4])
        conv6 = Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(up6)
        drop6 = Dropout(0.1)(conv6)
        conv6 = Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop6)

        up7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv6)
        up7 = concatenate([up7, conv3])
        conv7 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(up7)
        drop7 = Dropout(0.1)(conv7)
        conv7 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop7)

        up8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv7)
        up8 = concatenate([up8, conv2])
        conv8 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(up8)
        drop8 = Dropout(0.1)(conv8)
        conv8 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop8)

        up9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8)
        up9 = concatenate([up9, conv1])
        conv9 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(up9)
        drop9 = Dropout(0.1)(conv9)
        conv9 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(drop9)

        outputs = Conv2D(1, (1,1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mean_iou'])
        # model.summary()

        stop_run = EarlyStopping(patience=3, verbose=1)
        checkpoint_model = ModelCheckpoint('my_unet.h5', verbose=1, save_best_only=True)

        results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=10, callbacks=[stop_run, checkpoint_model])

        model.save('my_unet2.h5')

    return results


if __name__ == '__main__':
    fit_model(X_train, y_train)

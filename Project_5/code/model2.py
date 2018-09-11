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

def object_mean_iou(y_labeled_true, y_labeled_pred):
    num_y_labeled_true = y_labeled_true.max()
    num_y_labeled_pred = y_labeled_pred.max()
    threshold_ious = []
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        true_positives = 0
        false_negative = 0
        pred_obj_preload = []
        pred_obj_size = []
        for predicted_object_id in range(1, num_y_labeled_pred + 1):
            pred_obj = y_labeled_pred == predicted_object_id
            pred_obj_preload.append(pred_obj)
            pred_obj_size.append(np.count_nonzero(pred_obj))
        # a true positive for given threshold is when a _single_ object in the prediction corresponds to a given true object
        for true_object_id in range(1, num_y_labeled_true + 1):
            true_obj = y_labeled_true == true_object_id
            true_obj_size = np.count_nonzero(true_obj)
            matches = 0
            for predicted_object_id in range(1, num_y_labeled_pred + 1):
                # calculate the iou for this object and the true object
                this_pred_obj = pred_obj_preload[predicted_object_id-1]
                this_pred_obj_size = pred_obj_size[predicted_object_id-1]
                intersection = np.count_nonzero(true_obj & this_pred_obj)
                union = true_obj_size + this_pred_obj_size - intersection
                iou = intersection / union
                if iou > threshold:
                    matches += 1
            if matches == 1:
                true_positives += 1
            if matches == 0:
                false_negative += 1
        false_positive = num_y_labeled_pred - true_positives
        threshold_ious.append(true_positives / (true_positives + false_positive + false_negative))
    return sum(threshold_ious) / len(threshold_ious)


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

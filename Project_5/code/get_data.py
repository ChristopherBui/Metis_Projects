import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2
import pickle

import matplotlib.pyplot as plt

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Note: This script is for acquiring the training and validation data if you have
# downloaded the image files from kaggle and would like to work with those.
# main.py will load the data from pickle files that are already processed

# ignore warning messages
warnings.filterwarnings('ignore')

# set the dimensions of the images
width = 128
height= 128
channels= 3
train_path = '/stage1_train/'
test_path = '/stage1_test/'

# get id's of folders holding each image
train_ids = os.listdir(train_path)
test_ids = os.listdir(test_path)

# set the dimensions of X_train list
X_train = np.zeros((len(train_ids), height, width, channels), dtype=np.uint8)
y_train = np.zeros((len(train_ids), height, width, 1), dtype=np.bool)

def get_data():

    # get training data
    for n, train_id in enumerate(train_ids):
        root_path = os.path.join(train_path + train_id)

        img_path = os.path.join(root_path + '/images/')
        mask_path = os.path.join(root_path + '/masks/')

        img = imread(img_path + train_id + '.png', as_gray=True)[:,:,:channels]

        img = resize(img, (height, width), preserve_range=True, mode='constant').astype(np.uint8)
        #img = np.expand_dims(img, axis=0)

        X_train[n] = img

        mask_placeholder = np.zeros((width, height, 1), dtype=np.bool)

        for mask in os.listdir(mask_path):
            mask_ = imread(mask_path + mask)
            mask_ = resize(mask_, (height, width), preserve_range=True, mode='constant')
            mask_ = np.expand_dims(mask_, axis=-1)

            mask_placeholder = np.maximum(mask_placeholder, mask_)

        y_train[n] = mask_placeholder

    # get testing data
    X_test = np.zeros((len(test_ids), height, width, channels), dtype=np.uint8)

    for n, test_id in enumerate(test_ids):
        root_path = os.path.join(test_path + test_id)

        img_path = os.path.join(root_path + '/images/')

        img = imread(img_path + test_id + '.png', as_gray=True)[:,:,:channels]
        img = resize(img, (height, width), preserve_range=True, mode='constant').astype(np.uint8)

        X_test[n] = img

    # create training set & validation set
    X_validation = X_train[600:]
    X_train = X_train[:600]

    y_validation = y_train[600:]
    y_train = y_train[:600]

    # pickle the data
    with open('X_train.pickle','wb') as f:
        pickle.dump(X_train, f)

    with open('y_train.pickle','wb') as f:
        pickle.dump(y_train, f)

    with open('X_validation.pickle','wb') as f:
        pickle.dump(X_validation, f)

    with open('y_validaiton.pickle','wb') as f:
        pickle.dump(y_validaiton, f)

    with open('X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)

    return X_train, y_train, X_validaiton, y_validation, X_test


if __name__ == '__main__':
    get_data()

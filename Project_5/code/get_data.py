import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2
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

# ignore warning messages
warnings.filterwarnings('ignore')

# set random seed
seed = 24
random.seed = seed
np.random.seed = seed

# set the dimensions of the images
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/Users/christopher/Desktop/Project_5/nucleus/stage1_train/'
TEST_PATH = '/Users/christopher/Desktop/Project_5/nucleus/stage1_test/'

# get id's of folders holding each image
train_ids = os.listdir(TRAIN_PATH)
test_ids = os.listdir(TEST_PATH)

# set the dimensions of X_train list
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

def get_data():
    # get training data
    for n, train_id in enumerate(train_ids):
        root_path = os.path.join(TRAIN_PATH + train_id)

        img_path = os.path.join(root_path + '/images/')
        mask_path = os.path.join(root_path + '/masks/')


        img = imread(img_path + train_id + '.png', as_gray=True)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, mode='constant').astype(np.uint8)
        #img = np.expand_dims(img, axis=0)

        X_train[n] = img

        mask_placeholder = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)

        for mask in os.listdir(mask_path):
            mask_ = imread(mask_path + mask)
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, mode='constant')
            mask_ = np.expand_dims(mask_, axis=-1)

            mask_placeholder = np.maximum(mask_placeholder, mask_)

        y_train[n] = mask_placeholder

    # get testing data
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    for n, test_id in enumerate(test_ids):
        root_path = os.path.join(TEST_PATH + test_id)

        img_path = os.path.join(root_path + '/images/')

        img = imread(img_path + test_id + '.png', as_gray=True)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, mode='constant').astype(np.uint8)

        X_test[n] = img


    # pickle the data
    with open('X_train.pickle','wb') as f:
        pickle.dump(X_train, f)

    with open('y_train.pickle','wb') as f:
        pickle.dump(y_train, f)

    with open('X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)

    return X_train, y_train, X_test

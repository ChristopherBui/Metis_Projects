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

from get_data import *
from model import *


# ignore warning messages
warnings.filterwarnings('ignore')


def iou(img1, img2):
    intersection = np.sum(img1 * img2)
    union = np.sum(img1+img2) - intersection

    return intersection / union


def main():

    # get the data
    root = os.getcwd()
    path = root[:len(root)-len('code')]+'data/'

    X_train = pickle.load(open(path + 'X_train.pickle','rb'))
    y_train = pickle.load(open(path + 'y_train.pickle','rb'))
    X_validation = pickle.load(open(path + 'X_validaiton.pickle','rb'))
    y_validation = pickle.load(open(path + 'y_validation.pickle','rb'))

    # instantiate U-NET model & fit to training data
    model_ = fit_model(X_train, y_train)

    # predict on validation set
    y_pred = model_.predict(X_validation, verbose=1)

    with open('y_pred.pickle','wb') as f:
        pickle.dump(y_pred,f)

    # compute Intersection over Union values
    iou_values = []

    for i in range(len(y_validation)):
        iou_ = iou(y_true[i], y_pred[i])
        iou_values.append(iou_)

    average_iou = np.mean(iou_values)


    print('average_iou: ', average_iou)


    return y_pred, iou_values, average_iou



if __name__ == '__main__':
    main()

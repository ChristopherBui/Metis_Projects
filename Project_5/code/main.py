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
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# ignore warning messages
warnings.filterwarnings('ignore')

from get_data import *
from model import *


def main():

    # get the data
    X_train, y_train, X_test = get_data()

    # instantiate U-NET model & fit to training data
    model = fit_model(X_train, y_train)

    # predict
    # pred_validation = model.predict(








if __name__ == '__main__':
    main()

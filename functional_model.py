import os, logging
from subprocess import call
import tensorflow as tf
import numpy as np
import load_data
from utils import parse_args, dir_utils
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

#gettingg rid of "does not support AVX" warnings and info logs
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#CONST VARIABLES

class Naive_MLP(object):
    def __init__(self, data, batch_size=10, activations='relu',
            optimizer='Adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy']):
        '''
        - data: the object returned from load data
        '''
        self.data = data
        self.batch_size = batch_size
        self.activations = activations
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def splittingData(self):



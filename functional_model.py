import os, logging
from subprocess import call
import tensorflow as tf
import numpy as np
import load_data
from utils import parse_args, dir_utils
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#getting rid of "does not support AVX" warnings and info logs
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#CONST VARIABLES

class Naive_MLP(object):
    def __init__(self, data):
        '''
        - data: the object returned from load data
        '''
        self.data = data

    #splitting data into the right format of [[obs_vec], [acts_vec]]
    def get_formatted_data(self, data_type = 'train'):
        obs_vec = []
        act_vec = []
        if data_type == 'train':
            obs_vec, act_vec = self.data.generator('train')
        elif data_type == 'validation':
            obs_vec, act_vec = self.data.generator('validation')
        elif data_type == 'test':
            obs_vec, act_vec = self.data.generator('test')
        else:
            raise ValueError('Invalid data type')

        return obs_vec, act_vec

    '''Create a model'''
    def create_model(self, activations=['relu','softmax'], num_hidden_nodes = [256,128,64,20]):
        train_obs_vec, train_act_vec = self.get_formatted_data('train')
        
        input_layer = Input(shape=(len(train_obs_vec),))
        hidden_layer = Dense(num_hidden_nodes[0], activations=activations[0])(input_layer)
        for i in range(1,len(num_hidden_nodes)-1):
            hidden_layer = Dense(num_hidden_nodes[i], activations=activations[0])(hidden_layer)

        output_layer = Dense(num_hidden_nodes[len(num_hidden_nodes)-1], activations = activations[len(activations)-1])
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    '''train the model we just created'''
    def train_model(self, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
            loss ='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            epochs=100,
            batch_size=10):
        #get the training data and validation data
        train_obs_vec, train_act_vec = self.get_formatted_data('train')
        validation_obs_vec, validation_act_vec = self.get_formatted_data('validation')
        test_obs_vec, test_act_vec = self.get_formatted_data('test')

        #obtain input and output layer to create the model
        model = self.create_model
        model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
        tr_history = model.fit(train_obs_vec, train_act_vec,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(validation_obs_vec,validation_act_vec),
                shuffle = True)
        model.evaluate(test_obs_vec, test_act_vec)
        return model

import os, logging
from subprocess import call
import tensorflow as tf
import numpy as np
import load_data
from utils import parse_args, dir_utils
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

#getting rid of "does not support AVX" warnings and info logs
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#CONST VARIABLES
DATAPATH = os.path.dirname(os.path.realpath(__file__))

#load data
#call("python utils/parse_args.py --datapath " + DATAPATH + "/data/Hanabi-Full_2_6_150.pkl", shell=True)
class Naive_MLP (object):
    def __init__(self, learning_rate=0.001, ):
        self.learning_rate = learning_rate

    def main(self):
        #parse arguments
        args = parse_args.parse()
        args.datapath = DATAPATH + "/data/Hanabi-Full_2_6_150.pkl"
        args = parse_args.resolve_datapath(args)

        #create/load data
        '''
        - data: a reference to the Dataset object (refer to load_data.py)
        '''
        data = load_data.main(args)
        #getting train_data info
        [_,_,train_agent_obs], train_agent_act = data.generator('train')




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
DATAPATH = os.path.dirname(os.path.realpath(__file__))

#load data
#call("python utils/parse_args.py --datapath " + DATAPATH + "/data/Hanabi-Full_2_6_150.pkl", shell=True)

def main():
    #parse arguments
    args = parse_args.parse()
    args.datapath = DATAPATH + "/data/Hanabi-Full_2_6_150.pkl"
    args = parse_args.resolve_datapath(args)

    #create/load data
    '''
    - data: a reference to the Dataset object (refer to load_data.py)
     '''
    data = load_data.main(args)


if __name__ == "__main__":
    main()

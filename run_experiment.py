data             hanabi-env           __pycache__    train.pyc
evaluate.py      load_data.py         README.md      utils
huyen29@jam-MS-7B45:~/ganabi$ git checkout
M	hanabi-env/hanabi_lib/CMakeFiles/CMakeDirectoryInformation.cmake
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/CXX.includecache
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/DependInfo.cmake
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/build.make
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.o
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/depend.internal
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/flags.make
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.o
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.o
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.o
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.o
M	hanabi-env/hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.o
M	hanabi-env/hanabi_lib/Makefile
M	hanabi-env/hanabi_lib/cmake_install.cmake
M	hanabi-env/hanabi_lib/libhanabi.a
M	load_data.py
Your branch is up to date with 'origin/Sequential-Model'.
huyen29@jam-MS-7B45:~/ganabi$ vim functional_model.py 
huyen29@jam-MS-7B45:~/ganabi$ vim functional_model.py 
huyen29@jam-MS-7B45:~/ganabi$ clear

huyen29@jam-MS-7B45:~/ganabi$ vim functional_model.py 
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


                                                              37,0-1        Bot


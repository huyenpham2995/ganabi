from utils import parse_args
import importlib
import load_data
import gin
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

@gin.configurable
class Trainer(object):
    @gin.configurable
    def __init__(self, args,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 batch_size=None,
                 epochs=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

def set_up_vars():
    activations = ['relu', 'softmax']
    num_hidden_nodes = [256,128,64,20]
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    loss ='sparse_categorical_crossentropy'
    metrics=['accuracy']
    epochs=100
    batch_size=32
    
    return activations, num_hidden_nodes, optimizer, loss, metrics, epochs, batch_size

def main(data, args):
    '''
    trainer = Trainer(args) # gin configured

    #FIXME: combine into one line once stuff works
    mode_module = importlib.import_module(args.mode)                          
    model = mode_module.build_model(args)

    model.compile(
            optimizer = trainer.optimizer,
            loss = trainer.loss,
            metrics = trainer.metrics)

    tr_history = model.fit_generator(
            generator = data.generator('train'),
            verbose = 2, # one line per epoch
            batch_size = trainer.batch_size, 
            epochs = trainer.epochs, # = total data / batch_size
            validation_split = 0.1, # fraction of data used for val
            shuffle = True)
    '''
    #get the training data and validation data
    activations, num_hidden_nodes, optimizer, loss, metrics, epochs, batch_size = set_up_vars()
    train_obs, train_act = data.generator(batch_type='train')
    validation_obs, validation_act= data.generator(batch_type='validation')
    test_obs, test_act = data.generator(batch_type='test')

    #import pdb; pdb.set_trace()
    input_layer = Input(shape=([len(train_obs[0])]))
    hidden_layer = Dense(num_hidden_nodes[0], activation=activations[0])(input_layer)
    for i in range(1,len(num_hidden_nodes)-1):
        hidden_layer = Dense(num_hidden_nodes[i], activation=activations[0])(hidden_layer)

    output_layer = Dense(num_hidden_nodes[len(num_hidden_nodes)-1], activation = activations[len(activations)-1])(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=optimizer,
            loss=loss,
            metrics=metrics)
    tr_history = model.fit(train_obs[0],train_act[0],
            #batch_size = batch_size,
            epochs=epochs,
            verbose=1)
            #validation_data=(),
            
        #model.evaluate(test_obs_vec, test_act_vec)


    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)

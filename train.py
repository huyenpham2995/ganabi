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
    loss ='mean_squared_error'
    metrics=['accuracy']
    epochs=100
    batch_size=32
    
    return activations, num_hidden_nodes, optimizer, loss, metrics, epochs, batch_size

def main(data, args):
    #get the training data and validation data
    activations, num_hidden_nodes, optimizer, loss, metrics, epochs, batch_size = set_up_vars()
    train_obs, train_act = data.generator(batch_type='train')
    validation_obs, validation_act= data.generator(batch_type='validation')
    test_obs, test_act = data.generator(batch_type='test')
    
    # creating layers for model and linking them
    input_layer = Input(shape=(len(train_obs[0]),))
    hidden_layer = Dense(num_hidden_nodes[0], activation=activations[0])(input_layer)
    for i in range(1,len(num_hidden_nodes)-1):
        hidden_layer = Dense(num_hidden_nodes[i], activation=activations[0])(hidden_layer)
    flatten_layer = Flatten()(hidden_layer)
    output_layer = Dense(num_hidden_nodes[len(num_hidden_nodes)-1], activation = activations[len(activations)-1])(flatten_layer)
    model = Model(inputs=input_layer, outputs=output_layer)


    # compiling model
    model.compile(optimizer=optimizer,
            loss=loss,
            metrics=metrics)

    # training model
    tr_history = model.fit(train_obs,train_act,
            epochs=epochs,
            verbose=1)

    return model, test_obs, test_act

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)

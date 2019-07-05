import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

data = load_data.main("/data/Hanabi-Full_2_6_150.pkl")

train_x = np.array([0, 0, 0, 0, 0, 0], dtype = int)
train_y = np.array([1, 1, 1, 1, 1, 1], dtype = int)

test_x = train_x
test_y = train_y

num_inputs = 1
num_hidden_nodes = 64
num_outputs = 2
batch_size = 10
num_epochs = 1
learning_rate = 0.001

inputs = Input(shape=([num_inputs]))
hidden_layer = Dense(num_hidden_nodes, activation='relu')(inputs)
hidden_layer = Dense(num_hidden_nodes, activation='relu')(hidden_layer)
predictions = Dense(batch_size, activation='softmax')(hidden_layer)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, num_epochs)
model.evaluate(test_x, test_y)

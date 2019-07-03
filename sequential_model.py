
# loading data
import load_data
data = load_data.main("/data/Hanabi-Full_2_6_150.pkl")

# TODO: have actual values
num_inputs = 658
num_hidden_nodes = 64
num_outputs = 25
batch_size = 10
num_epochs = 1
learning_rate = 0.001

# create layers
input_layer = tf.keras.layers.Flatten(num_inputs, 1)
hidden_layer_1 = tf.keras.layers.Dense(units=num_hidden_nodes, activation='relu')
hidden_layer_2 = tf.keras.layers.Dense(units=num_hidden_nodes, activation='relu')
output_layer = tf.keras.layers.Dense(units=num_outpus, activation = 'softmax')

# instantiate model
model = tf.keras.models.Sequental([input_layer, hidden_layer_1, hidden_layer_2, output_layer])

# compile model and set loss and optimizer func
model.compile(loss='sparse_categorical_crossentropy',
               optimizer=tf.keras.optimizer.Adam(learning_rate),
               metrics=['accuracy'])

# train model
model.fit(train_x, train_y, num_epochs)

# evaluate/test model
model.evaluate(test_x, test_y)

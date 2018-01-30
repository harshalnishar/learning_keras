import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import os.path

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter = ',')
X = dataset[:, 0:8]
Y = dataset[:,8]

if not os.path.isfile('dnn_model.json'):
    # Model defition:
    # Input layer  : 8 nodes
    # Hidden layer1: 15 nodes
    # Hidden layer2: 10 nodes
    # Output layer : 1 node (Binary classification)
    dnn_model = Sequential()
    dnn_model.add(Dense(15, input_dim = 8, activation = 'relu'))
    dnn_model.add(Dense(10, activation = 'relu'))
    dnn_model.add(Dense(1, activation = 'sigmoid'))

    dnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Training
    dnn_model.fit(X, Y, epochs = 1000, batch_size = 10)
    print(dnn_model.evaluate(X, Y))

    # Save model
    json_string = dnn_model.to_json()
    json_file = open('dnn_model.json', 'w')
    json_file.write(json_string)
    json_file.close()

    dnn_model.save_weights('dnn_weights.hdf5')

json_file = open('dnn_model.json', 'r')
json_string = json_file.read()
json_file.close()
dnn_model = model_from_json(json_string)
dnn_model.load_weights('dnn_weights.hdf5')

predict = dnn_model.predict(np.reshape(X[1, :],(1,8)))

print(predict)

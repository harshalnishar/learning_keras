import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter = ',')
X = dataset[:, 0:8]
Y = dataset[:,8]
print(X.shape)

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

dnn_model.fit(X, Y, epochs = 300, batch_size = 10)

print(dnn_model.evaluate(X, Y))

predict = dnn_model.predict(np.reshape(X[1, :],(1,8)))

print(predict)

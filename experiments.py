import numpy as np
import nnfs
from nnfs.datasets import vertical_data

from dense_layer import DenseLayer
from activation import ReLUActivation, SoftmaxActivation
from loss import CategoricalCrossEntropyLoss

X, y = vertical_data(samples=100, classes=3) 

dense1 = DenseLayer(2, 3)
activation1 = ReLUActivation()
dense2 = DenseLayer(3, 3)
activation2 = SoftmaxActivation()

loss_function = CategoricalCrossEntropyLoss()

lowest_loss = 9999999

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

perturb = lambda n, m: 0.05 * np.random.randn(n, m)

for iteration in range(10000):

    dense1.weights += perturb(2, 3)    
    dense1.biases += perturb(1, 3)    
    dense2.weights += perturb(3, 3)    
    dense2.biases += perturb(1, 3)    

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    if loss < lowest_loss:
        print(f'New set of weights found, iteration: {iteration} loss: {loss} acc: {accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

import numpy as np
import nnfs
from nnfs.datasets import vertical_data

from dense_layer import DenseLayer
from activation import ReLUActivation, SoftmaxActivation
from loss import CategoricalCrossEntropyLoss
from classifiers import SoftmaxClassifier

X, y = vertical_data(samples=100, classes=3) 

dense1 = DenseLayer(2, 3)
activation1 = ReLUActivation()
dense2 = DenseLayer(3, 3)
loss_activation = SoftmaxClassifier()

loss_function = CategoricalCrossEntropyLoss()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y)
print(loss_activation.output[:5])
print('loss:', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print('acc:', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

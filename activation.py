import numpy as np

class ReLUActivation:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class SoftmaxActivation:
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilites

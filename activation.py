import numpy as np


class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0


class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilites

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        for idx, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            self.dinputs[idx] = np.dot(jacobian, single_dvalues)

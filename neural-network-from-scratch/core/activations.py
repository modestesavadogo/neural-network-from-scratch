import numpy as np


class ReLU:
    """
    Rectified Linear Unit activation function.

    Forward:  f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    """
    Softmax activation function — maps raw scores to a probability distribution.

    Forward:  softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
              (subtracting max for numerical stability)
    Backward: combined with CrossEntropy loss for efficiency (see losses.py)
    """

    def forward(self, inputs):
        # Subtract max per sample for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian, single_dvalues)

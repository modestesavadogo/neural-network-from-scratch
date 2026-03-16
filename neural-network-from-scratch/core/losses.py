import numpy as np
from .activations import Softmax


class Loss:
    """Base class — computes the mean loss over a batch."""

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

    def regularization_loss(self, layer):
        """Compute the L1 + L2 penalty for a given layer."""
        reg_loss = 0.0
        if layer.l1_reg > 0:
            reg_loss += layer.l1_reg * np.sum(np.abs(layer.weights))
        if layer.l2_reg > 0:
            reg_loss += layer.l2_reg * np.sum(layer.weights ** 2)
        return reg_loss


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss.

    Works with both sparse labels  (shape: [N])
    and one-hot encoded labels     (shape: [N, C]).

    Forward:  L = -sum( y_true * log(y_pred) )
    Backward: dL/dy_pred = -y_true / y_pred  (normalised by batch size)
    """

    def forward(self, y_pred, y_true):
        n = len(y_pred)
        # Clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if y_true.ndim == 1:          # sparse
            correct_confidences = y_pred_clipped[range(n), y_true]
        else:                          # one-hot
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        n      = len(dvalues)
        labels = len(dvalues[0])

        if y_true.ndim == 1:
            y_true = np.eye(labels)[y_true]   # convert to one-hot

        self.dinputs = (-y_true / dvalues) / n


class SoftmaxWithCrossEntropy:
    """
    Fused Softmax + Categorical Cross-Entropy layer.

    Combining the two gives a much simpler backward pass:
        dL/dz = y_pred - y_true   (after normalising by batch size)

    Use this as the *output* layer of any classification network.
    """

    def __init__(self):
        self.activation = Softmax()
        self.loss        = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        n = len(dvalues)

        if y_true.ndim == 2:               # one-hot → sparse
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(n), y_true] -= 1   # subtract 1 at the true class
        self.dinputs /= n                      # normalise

    def regularization_loss(self, layer):
        return self.loss.regularization_loss(layer)

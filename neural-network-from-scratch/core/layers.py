import numpy as np


class DenseLayer:
    """
    Fully connected (dense) layer.

    Parameters
    ----------
    n_inputs   : number of input features
    n_neurons  : number of neurons in this layer
    l1_reg     : L1 regularisation strength (promotes sparsity)
    l2_reg     : L2 regularisation strength (penalises large weights)

    Attributes after forward()
    --------------------------
    output     : shape (batch, n_neurons)

    Attributes after backward()
    ---------------------------
    dweights   : gradient w.r.t. weights
    dbiases    : gradient w.r.t. biases
    dinputs    : gradient w.r.t. inputs  (passed to the previous layer)
    """

    def __init__(self, n_inputs, n_neurons, l1_reg=0.0, l2_reg=0.0):
        # Small random weights, biases start at zero
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        self.l1_reg  = l1_reg
        self.l2_reg  = l2_reg

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)

        # L1 regularisation gradient: sign(w)
        if self.l1_reg > 0:
            self.dweights += self.l1_reg * np.sign(self.weights)

        # L2 regularisation gradient: 2 * lambda * w
        if self.l2_reg > 0:
            self.dweights += 2 * self.l2_reg * self.weights

        # Gradient passed to the previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)


class DropoutLayer:
    """
    Inverted dropout layer.

    During training a random fraction of neurons is zeroed out each forward
    pass.  The remaining activations are scaled up by 1/(1-rate) so that the
    expected value is preserved — no rescaling is needed at inference time.

    Parameters
    ----------
    drop_rate : fraction of neurons to drop  (e.g. 0.2 drops 20 %)
    """

    def __init__(self, drop_rate):
        self.rate = 1 - drop_rate   # success (keep) rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        # Binomial mask, scaled so expected value stays the same
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

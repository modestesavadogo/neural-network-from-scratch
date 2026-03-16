"""
neural-network-from-scratch · core
===================================
A minimal neural-network library built with NumPy only.

Quick start
-----------
>>> from core import DenseLayer, ReLU, SoftmaxWithCrossEntropy, Adam
"""

from .layers      import DenseLayer, DropoutLayer
from .activations import ReLU, Softmax
from .losses      import CategoricalCrossEntropy, SoftmaxWithCrossEntropy
from .optimizers  import SGD, Adagrad, RMSProp, Adam

__all__ = [
    "DenseLayer", "DropoutLayer",
    "ReLU", "Softmax",
    "CategoricalCrossEntropy", "SoftmaxWithCrossEntropy",
    "SGD", "Adagrad", "RMSProp", "Adam",
]

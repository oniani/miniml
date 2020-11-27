# type: ignore

import numpy as np

import miniml.tensor as T

from abc import ABC, abstractmethod


class Layer(ABC):
    """Abstract class for a layer of the neural network."""

    @abstractmethod
    def __init__(self):
        """Initilize the variables."""

        pass

    @abstractmethod
    def forward(self, x: T.Tensor):
        """Perform a forward pass."""

        pass

    @abstractmethod
    def backward(self, dF: T.Tensor):
        """Perform a backpropagation."""

        pass

    def optimize(self):
        """Perform an optimization step."""

        pass


class Linear(Layer):
    """A linear layer."""

    def __init__(self, in_dim: int, out_dim: int, gpu: bool = False) -> None:
        """Initilize variables."""

        self._weights: T.Tensor = T.Random.rand((in_dim, out_dim), gpu)
        self._biases: T.Tensor = T.Tensor.rand(out_dim, gpu)
        self._gpu: bool = gpu

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        self._prev: T.Tensor = x
        return T.Tensor.dot(self._weights, x) + self._bias

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation."""

        # Derivative of the cost function w.r.t. W
        self._dW: T.Tensor = T.Tensor.dot(dF, self._prev.T)

        # Derivative of the cost function w.r.t. b
        self._db: T.Tensor = T.Tensor(np.mean(dF, axis=1, keepdims=True))

        # Apply the chain rule
        return T.Tensor.dot(self._weights.T, dF)

    def optimize(self, lr: float = 1e-3) -> None:
        """Perform an optimization step."""

        self._weights -= self._dW * lr
        self._biases -= self._db * lr

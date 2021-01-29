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

    def __init__(self, in_dim: int, out_dim: int, gpu: bool = True) -> None:
        """Initilize variables."""

        self._gpu: bool = gpu
        self._weights: T.Tensor = T.Random.rand((out_dim, in_dim), self._gpu)
        self._biases: T.Tensor = T.Random.rand((out_dim, in_dim), self._gpu)
        self._type = "Linear"

    @property
    def type(self) -> str:
        """Return the type of the layer."""

        return self._type

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        self._prev: T.Tensor = x
        # TODO: Need to implement broadcasting on the GPU
        return T.Ops.dot(self._weights, x, gpu=True)  # + self._biases

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation."""

        # Derivative of the cost function w.r.t. W
        self._dW: T.Tensor = T.Ops.dot(dF, self._prev.T, gpu=True)

        # Derivative of the cost function w.r.t. b
        self._db: T.Tensor = T.Tensor(np.mean(dF, axis=1, keepdims=True), gpu=True)

        # Apply the chain rule
        return T.Ops.dot(self._weights.T, dF), self._dW, self._db

    def optimize(self, lr: float = 1e-3) -> None:
        """Perform an optimization step."""

        self._weights -= self._dW * lr
        self._biases -= self._db * lr

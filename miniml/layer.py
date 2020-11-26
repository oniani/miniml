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
        """Initilize variables.

        Parameters
        ----------
        in_dim  : Number of input dimensions.
        out_dim : Number of output dimensions.
        gpu     : Specifies whether the computations are performed on GPU.
        """

        self._weights: T.Tensor = T.Random.rand((in_dim, out_dim), gpu)
        self._biases: T.Tensor = T.Tensor.rand(out_dim, gpu)
        self._gpu: bool = gpu

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass.
        
        Parameters
        ----------
        x : Data (activated or not) of the previous layer
        """

        self._prev: T.Tensor = x
        return T.Tensor.dot(self._weights, x) + self._bias

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        Parameters
        ----------
        dF: upper layer gradient
        """

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


class ReLU(Layer):
    def __init__(self, output_dim: int) -> None:
        """Initilize variables."""

        self.units: int = output_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        self._prev: T.Tensor = T.Tensor(np.maximum(0, x.data))
        return self._prev

    def backward(self, dF: T.Tensor) -> None:
        """Perform a backpropagation.

        There are two cases:
           (1) if x <= 0, derivative is 0' = 0
           (2) if x >  0, derivative is x' = 1
        """

        return dF * T.Tensor(np.where(self._prev <= 0, 0, 1))


class Sigmoid(Layer):
    def __init__(self, output_dim: int) -> None:
        """Initilize variables."""

        self._units: int = output_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        self._prev: T.Tensor = T.Tensor(1 / (1 + np.exp(-x.data)))
        return self._prev

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        If the sigmoid function is denoted as s, it can be shown that the
        derivative is computed as s * (1 - s).
        """

        return dF * self._prev * (1 - self._prev)


class Tanh(Layer):
    def __init__(self, output_dim: int) -> None:
        """Initilize variables."""

        self._units: int = output_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        self._prev: T.Tensor = T.Tensor(np.tanh(x.data))
        return self._prev

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

        tanh'(x) = ((e^x + e^{-x})^2 - (e^x - e^{-x})^2) / (e^x + e^{-x})^2
                 = 1 - ((e^x - e^{-x}) / (e^x + e^{-x}))^2
                 = 1 - tanh^2(x)
        """

        return T.Tensor(dF * (1 - np.square(np.tanh(self._prev.data))))


class Softmax(Layer):
    def __init__(self, output_dim: int) -> None:
        """Initilize variables."""

        self._units: int = output_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        exp: np.ndarray = np.exp(x.data - np.max(x.data))
        elf._prev: Tensor = exp / np.sum(exp)
        return self._prev

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation."""

        exp: np.ndarray = np.exp(self._prev.data - np.max(self._prev.data))
        val: np.ndarray = exp / np.sum(exp)
        return T.Tensor(-np.outer(val, val) + np.diag(val.flatten()))

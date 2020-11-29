# type: ignore

"""
Author: David Oniani

Description:
    A collection of activation functions and their derivatives.
"""

import numpy as np

import miniml.tensor as T
import miniml.layer as L


class LeakyReLU:
    def __init__(self, out_dim: int = 3, alpha: float = 1e-2) -> None:
        """Initilize variables for the LeakyReLU activation function."""

        self._units: int = out_dim
        self._alpha: float = alpha

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass (numerically stable)."""

        self._saved: T.Tensor = T.Ops.where(x > 0, x, x * self._alpha)
        return self._saved

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        Gradient computation:
           (1) if x < 0, derivative is alpha
           (2) if x > 0, derivative is 1
           (3) if x = 0, derivative is not well-defined, but can treat as 1
        """

        return dF * T.Ops.where(self._saved < 0, self._alpha, 1)


class ReLU(L.Layer):
    def __init__(self, out_dim: int = 3) -> None:
        """Initilize variables for the ReLU activation function."""

        self._units: int = out_dim
        self.type = "ReLU"

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass (numerically stable)."""

        self._saved: T.Tensor = T.Ops.maximum(x, 0)
        return self._saved

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        Gradient computation:
           (1) if x < 0, derivative is 0
           (2) if x > 0, derivative is 1
           (3) if x = 0, derivative is not well-defined, but can treat as 1
        """

        return dF * T.Ops.where(self._saved < 0, 0, 1)


class Tanh:
    def __init__(self, out_dim: int = 3) -> None:
        """Initilize variables for the Tanh activation function."""

        self._units: int = out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass (numerically stable)."""

        self._saved: T.Tensor = T.Ops.tanh(x)
        return self._saved

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        Gradient computation:
            tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

            tanh'(x) = ((e^x + e^{-x})^2 - (e^x - e^{-x})^2) / (e^x + e^{-x})^2
                     = 1 - ((e^x - e^{-x}) / (e^x + e^{-x}))^2
                     = 1 - tanh^2(x)
        """

        return dF * (1 - T.Ops.square(self._saved))


class Sigmoid(L.Layer):
    def __init__(self, out_dim: int = 3) -> None:
        """Initilize variables for the Sigmoid activation function."""

        self._units: int = out_dim
        self.type = "ReLU"

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass (numerically stable)."""

        fst: T.Tensor = 1 / (1 + T.Ops.exp(-x))
        pexp: T.Tensor = T.Ops.exp(x)
        snd: T.Tensor = pexp / (1 + pexp)
        self._saved: T.Tensor = T.Ops.where(x >= 0, fst, snd)
        return self._saved

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation.

        Gradient computation:
            If the sigmoid function is denoted as s, it can be shown that the
            derivative is computed as s * (1 - s).
        """

        return dF * self._saved * (1 - self._saved)


class Softmax:
    def __init__(self, out_dim: int = 3) -> None:
        """Initilize variables for the Softmax activation function."""

        self._units: int = out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass (numerically stable)."""

        exp: T.Tensor = T.Ops.exp(x - T.Reduce.max(x))
        self._saved: T.Tensor = exp / T.Reduce.sum(exp)
        return self._saved

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation."""

        raise NotImplementedError


class LogSoftmax:
    def __init__(self, out_dim: int = 3) -> None:
        """Initilize variables for the LogSoftmax activation function."""

        self._units: int = out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass (numerically stable)."""

        exp: T.Tensor = T.Ops.exp(x - T.Reduce.max(x))
        self._saved: T.Tensor = T.Ops.log(exp / T.Reduce.sum(exp))
        return self._saved

    def backward(self, dF: T.Tensor) -> T.Tensor:
        """Perform a backpropagation."""

        raise NotImplementedError

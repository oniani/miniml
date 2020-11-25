# type: ignore
"""
Author: David Oniani

A collection of activation functions and their derivatives. All functions are
implemented using in-place changes for the efficiency and speed.
"""

import numpy as np
import miniml.tensor as T


class A:
    """A collection of activation functions."""

    @staticmethod
    def leaky_relu(z: T.Tensor, alpha: float = 1e-2) -> T.Tensor:
        """The leaky rectified linear activation function (numerically
           stable).
        """

        return ((z > 0) * z) + ((z <= 0) * z * alpha)

    @staticmethod
    def relu(z: T.Tensor) -> T.Tensor:
        """The rectified linear activation function (numerically stable)."""

        return T.Ops.maximum(z, 0)

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """The tanh activation function (numerically stable)."""

        return T.Ops.tanh(z)

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """The sigmoid activation function (numerically stable)."""

        if (z >= 0).all():
            return 1 / (1 + T.Ops.exp(-z))

        return 1 / (1 + T.Ops.exp(z))

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """The softmax activation function (numerically stable)."""

        exp: T.Tensor = T.Ops.exp(z - T.Reduce.max(z))
        return exp / T.Reduce.sum(exp)

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """The log softmax activation function (numerically stable)."""

        exp: T.Tensor = T.Ops.exp(z - T.Reduce.max(z))
        return T.Ops.log(exp / T.Reduce.sum(exp))


class AD:
    """A collection of activation function derivatives."""

    @staticmethod
    def leaky_relu(z: T.Tensor, alpha: float = 0.01) -> T.Tensor:
        """Derivative of the leaky rectified linear activation function.
        
        There are three cases:
           (1) if x < 0, derivative is x * alpha
           (2) if x > 0, derivative is 1
           (3) if x = 0, derivative is not well-defined, but can treat as 1
        """

        return T.Ops.where(z < 0, z * alpha, 1)

    @staticmethod
    def relu(z: T.Tensor) -> T.Tensor:
        """Derivative of the rectified linear activation function.
        
        There are three cases:
           (1) if x < 0, derivative is 0
           (2) if x > 0, derivative is 1
           (3) if x = 0, derivative is not well-defined, but can treat as 1
        """

        return T.Ops.where(z < 0, 0, 1)

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """Derivative of the tanh activation function.
        
        tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

        tanh'(x) = ((e^x + e^{-x})^2 - (e^x - e^{-x})^2) / (e^x + e^{-x})^2
                 = 1 - ((e^x - e^{-x}) / (e^x + e^{-x}))^2
                 = 1 - tanh^2(x)
        """

        return 1 - T.Ops.square(T.Ops.tanh(z))

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """Derivative of the sigmoid activation function.
        
        If the sigmoid function is denoted as s, it can be shown that the
        derivative is computed as s * (1 - s).
        """

        if (z >= 0).all():
            s = 1 / (1 + T.Ops.exp(-z))
        else:
            s = 1 / (1 + T.Ops.exp(z))

        return s * (1 - s)

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the softmax activation function."""

        raise NotImplementedError

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the log softmax activation function."""

        raise NotImplementedError

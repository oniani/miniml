# type: ignore
"""
Author: David Oniani

A collection of activation functions and their derivatives. All functions are
implemented using in-place changes for the efficiency and speed.
"""

import numpy as np
import tensor as T


class Activation:
    """A collection of activation functions."""

    @staticmethod
    def relu(z: T.Tensor) -> T.Tensor:
        """Rectified linear activation function."""

        z.data: np.ndarray = np.maximum(0, z.data)
        return z

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """A tanh activation function."""

        z.data: np.ndarray = np.tanh(z.data)
        return z

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """A sigmoid activation function."""

        z.data: np.ndarray = 1 / (1 + np.exp(-z.data))
        return z

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """A softmax activation function."""

        exp: np.ndarray = np.exp(z.data)
        z.data: np.ndarray = exp / np.sum(exp)
        return z

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """A log softmax activation function."""

        exp: np.ndarray = np.exp(z.data)
        z.data: np.ndarray = np.log(exp / np.sum(exp))
        return z


class ActivationDerivative:
    """A collection of activation function derivatives."""

    @staticmethod
    def relu(z: T.Tensor) -> T.Tensor:
        """Derivative of the rectified linear activation function.
        
        There are two cases:
           (1) if x <= 0, derivative is 0' = 0
           (2) if x >  0, derivative is x' = 1
        """

        z.data: np.ndarray = np.where(z.data <= 0, 0, 1)
        return z

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """Derivative of the tanh activation function.
        
        tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

        tanh'(x) = ((e^x + e^{-x})^2 - (e^x - e^{-x})^2) / (e^x + e^{-x})^2
                 = 1 - ((e^x - e^{-x}) / (e^x + e^{-x}))^2
                 = 1 - tanh^2(x)
        """

        z.data: np.ndarray = 1 - np.square(np.tanh(z.data))
        return z

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """Derivative of the sigmoid activation function.
        
        If the sigmoid function is denoted as s, it can be shown that the
        derivative is computed as s * (1 - s).
        """

        s: np.ndarray = 1 / (1 + np.exp(-z.data))
        z.data: np.ndarray = s * (1 - s)
        return z

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the softmax activation function."""

        exp: np.ndarray = np.exp(z.data)
        z.data: np.ndarray = exp / np.sum(exp)
        z.data = -np.outer(z.data, z.data) + np.diag(z.data.flatten())
        return z

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the log softmax activation function."""

        raise NotImplementedError

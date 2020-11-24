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
        """The rectified linear activation function. (numerically stable)."""

        return T.Tensor.maximum(z, T.Tensor.zeros_like(z))

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """The tanh activation function (numerically stable)."""

        return T.Tensor.tanh(z)

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """The sigmoid activation function (numerically stable)."""

        if (z >= T.Tensor.zeros_like(z)).all():
            return 1 / (1 + T.Tensor.exp(-z))

        return 1 / (1 + T.Tensor.exp(z))

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """The softmax activation function (numerically stable)."""

        exp: T.Tensor = T.Tensor.exp(z - T.Tensor.max(z))
        return exp / T.Tensor.sum(exp)

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """The log softmax activation function (numerically stable)."""

        exp: T.Tensor = T.Tensor.exp(z - T.Tensor.max(z))
        return T.Tensor.log(exp / T.Tensor.sum(exp))


class ActivationDerivative:
    """A collection of activation function derivatives."""

    @staticmethod
    def relu(z: T.Tensor) -> T.Tensor:
        """Derivative of the rectified linear activation function.
        
        There are two cases:
           (1) if x <= 0, derivative is 0' = 0
           (2) if x >  0, derivative is x' = 1
        """

        return T.Tensor.where(z <= 0, 0, 1)

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """Derivative of the tanh activation function.
        
        tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

        tanh'(x) = ((e^x + e^{-x})^2 - (e^x - e^{-x})^2) / (e^x + e^{-x})^2
                 = 1 - ((e^x - e^{-x}) / (e^x + e^{-x}))^2
                 = 1 - tanh^2(x)
        """

        return 1 - T.Tensor.square(T.Tensor.tanh(z))

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """Derivative of the sigmoid activation function.
        
        If the sigmoid function is denoted as s, it can be shown that the
        derivative is computed as s * (1 - s).
        """

        if (z >= T.Tensor.zeros_like(z)).all():
            s = 1 / (1 + T.Tensor.exp(-z))
        else:
            s = 1 / (1 + T.Tensor.exp(z))

        return s * (1 - s)

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the softmax activation function."""

        raise NotImplementedError

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the log softmax activation function."""

        raise NotImplementedError


if __name__ == "__main__":
    # Tensors
    t1 = T.Tensor([-1, 0, 2, 3], gpu=True)
    t2 = T.Tensor([-1, 0, 2, 3], gpu=False)

    # Activation functions
    assert Activation.relu(t1)
    assert Activation.relu(t2)

    assert Activation.tanh(t1)
    assert Activation.tanh(t2)

    assert Activation.sigmoid(t1)
    assert Activation.sigmoid(t2)

    assert Activation.softmax(t1)
    assert Activation.softmax(t2)

    assert Activation.log_softmax(t1)
    assert Activation.log_softmax(t2)

    print("Activation functions, success")

    # Activation function derivatives
    assert ActivationDerivative.relu(t1)
    assert ActivationDerivative.relu(t2)

    assert ActivationDerivative.tanh(t1)
    assert ActivationDerivative.tanh(t2)

    assert ActivationDerivative.sigmoid(t1)
    assert ActivationDerivative.sigmoid(t2)

    print("Derivatives of `softmax` and `log_softmax` left to implement")

    # print(ActivationDerivative.softmax(t1))
    # print(ActivationDerivative.softmax(t2))

    # print(Activation.log_softmax(t1))
    # print(Activation.log_softmax(t2))

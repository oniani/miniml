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


class ActivationDerivative:
    """A collection of activation function derivatives."""

    @staticmethod
    def leaky_relu(z: T.Tensor, alpha: float = 0.01) -> T.Tensor:
        """Derivative of the leaky rectified linear activation function.
        
        There are three cases:
           (1) if x < 0, derivative is x * alpha
           (2) if x > 0, derivative is 1
           (3) if x = 0, derivative is not well-defined, but can treat as
               x * alpha
        """

        return T.Ops.where(
            z <= 0, z * alpha, T.Ops.fill(z.shape, 1, gpu=z._gpu)
        )

    @staticmethod
    def relu(z: T.Tensor) -> T.Tensor:
        """Derivative of the rectified linear activation function.
        
        There are three cases:
           (1) if x < 0, derivative is 0
           (2) if x > 0, derivative is 1
           (3) if x = 0, derivative is not well-defined, but can treat as 0
        """

        return T.Ops.where(z <= 0, 0, 1)

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


if __name__ == "__main__":
    # Tensors
    t1 = T.Tensor([-1, 0, 2, 3], gpu=True)
    t2 = T.Tensor([-1, 0, 2, 3], gpu=False)

    # Activation functions
    assert Activation.leaky_relu(t1) == Activation.leaky_relu(t2).gpu()
    assert Activation.relu(t1) == Activation.relu(t2).gpu()
    assert Activation.tanh(t1) == Activation.tanh(t2).gpu()
    assert Activation.sigmoid(t1) == Activation.sigmoid(t2).gpu()
    assert Activation.softmax(t1) == Activation.softmax(t2).gpu()
    assert Activation.log_softmax(t1) == Activation.log_softmax(t2).gpu()

    print("Activation functions, success")

    # Activation function derivatives
    assert (
        ActivationDerivative.leaky_relu(t1)
        == ActivationDerivative.leaky_relu(t2).gpu()
    )
    assert ActivationDerivative.relu(t1) == ActivationDerivative.relu(t2).gpu()
    assert ActivationDerivative.tanh(t1) == ActivationDerivative.tanh(t2).gpu()
    assert (
        ActivationDerivative.sigmoid(t1)
        == ActivationDerivative.sigmoid(t2).gpu()
    )

    print("Derivatives of `softmax` and `log_softmax` left to implement")

    # print(ActivationDerivative.softmax(t1))
    # print(ActivationDerivative.softmax(t2))

    # print(Activation.log_softmax(t1))
    # print(Activation.log_softmax(t2))

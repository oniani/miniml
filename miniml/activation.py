# type: ignore
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
        """Derivative of the rectified linear activation function."""

        raise NotImplementedError

    @staticmethod
    def tanh(z: T.Tensor) -> T.Tensor:
        """Derivative of the tanh activation function."""

        t = Activation.tanh(z)
        return T.Tensor(1 - np.exp2(t.data))

    @staticmethod
    def sigmoid(z: T.Tensor) -> T.Tensor:
        """Derivative of the sigmoid activation function."""

        s = Activation.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the softmax activation function."""

        raise NotImplementedError

    @staticmethod
    def log_softmax(z: T.Tensor) -> T.Tensor:
        """Derivative of the log softmax activation function."""

        raise NotImplementedError

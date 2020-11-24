# type: ignore

import tensor as T

# TODO: Consider using an abstract class for loss functions as well


class MSE:
    """The means squared error loss function."""

    def __init__(self, pred: T.Tensor, real: T.Tensor):
        self._pred: T.Tensor = pred
        self._real: T.Tensor = real

    def forward(self):
        return T.Tensor(np.power((self._pred - self._real).data, 2).mean())

    def backward(self):
        return 2 * ((self._pre - self._real).data).mean()

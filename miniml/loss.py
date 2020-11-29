# type: ignore

import miniml.tensor as T

# TODO: Consider using an abstract class for loss functions as well


class MeanSquaredError:
    def __init__(self, pred: T.Tensor, real: T.Tensor):
        self._pred = pred
        self._real = real

    def forward(self):
        return T.Reduce(T.Ops.power(self._pred - self._real, 2))

    def backward(self):
        return 2 * T.Reduce.mean(self._pred - self._real)

# type: ignore

from typing import List

import numpy as np
import tensor as T
import activation as A


# Random seed for consistent results
np.random.seed(60)


class Net:
    """A test class."""

    _NN_ARCHITECTURE = [
        {"input_dim": 3, "output_dim": 6, "activation": "tanh"},
        {"input_dim": 6, "output_dim": 9, "activation": "tanh"},
        {"input_dim": 9, "output_dim": 6, "activation": "tanh"},
        {"input_dim": 6, "output_dim": 3, "activation": "log_softmax"},
    ]

    def __init__(self) -> None:
        """Initilize the network layers."""

        self._params = []
        for idx, layer in enumerate(self._NN_ARCHITECTURE):
            input_dim: int = layer["input_dim"]
            output_dim: int = layer["output_dim"]
            activation: str = layer["activation"]

            weights = T.Tensor(np.random.randn(input_dim, output_dim))
            biases = T.Tensor(np.random.randn(output_dim))

            if activation == "tanh":
                activation_fun = A.Activation.tanh

            elif activation == "sigmoid":
                activation_fun = A.Activation.sigmoid

            elif activation == "relu":
                activation_fun = A.Activation.relu

            elif activation == "softmax":
                activation_fun = A.Activation.softmax

            elif activation == "log_softmax":
                activation_fun = A.Activation.log_softmax

            else:
                raise RuntimeError("Invalid activation function.")

            self._params.append(
                {
                    "weights": weights,
                    "biases": biases,
                    "activation": activation_fun,
                }
            )

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        for layer in self._params:
            x = layer["activation"](
                T.Tensor.dot(x, layer["weights"]) + layer["biases"]
            )

        return x


if __name__ == "__main__":
    from pprint import pprint

    model = Net()
    t = T.Tensor(np.random.uniform(-100, 100, (2, 4, 3)))
    f = model.forward(t)
    pprint(f.data)

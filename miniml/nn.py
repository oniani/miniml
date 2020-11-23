# type: ignore

from typing import Callable, Dict, List, Union

import numpy as np
import tensor as T
import activation as A


class Net:
    """A test class."""

    _ACTIVATION_FUNCTION_TABLE: Dict[str, Callable[[T.Tensor], T.Tensor]] = {
        "tanh": A.Activation.tanh,
        "sigmoid": A.Activation.sigmoid,
        "relu": A.Activation.relu,
        "softmax": A.Activation.softmax,
        "log_softmax": A.Activation.log_softmax,
    }

    def __init__(self, nn_architecture: Dict[str, Union[int, str]]) -> None:
        """Initilize the network layers."""

        # The neural architecture provided by the user
        self._nn_architecture = nn_architecture

        # Network parameters
        self._params = []
        for layer in self._nn_architecture:
            # Get the data
            activation: str = layer["activation"]
            input_dim: int = layer["input_dim"]
            output_dim: int = layer["output_dim"]

            # Generate weights and biases, add the activation function
            fun: Dict[
                str, Callable[[T.Tensor], T.Tensor]
            ] = self._ACTIVATION_FUNCTION_TABLE.get(activation)
            weight: T.Tensor = T.Tensor(np.random.randn(input_dim, output_dim))
            bias: T.Tensor = T.Tensor(np.random.randn(output_dim))

            # Populate the parameters
            self._params.append({"A": fun, "W": weight, "b": bias})

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Perform a forward pass."""

        out: T.Tensor = x

        for layer in self._params:
            fun: Callable = layer["A"]
            out: T.Tensor = (
                fun(T.Tensor.dot(out, layer["W"]) + layer["b"])
                if fun
                else T.Tensor.dot(out, layer["W"]) + layer["b"]
            )

        return out


if __name__ == "__main__":
    # Some pretty-printing
    from pprint import pprint

    # Random seed for consistent results
    np.random.seed(57)

    # The architecture of the neural network. Field `output_dim` is not
    # required, but is retained for clarity
    nn_architecture: Dict[str, Union[int, str]] = [
        {"input_dim": 3, "output_dim": 6, "activation": "tanh"},
        {"input_dim": 6, "output_dim": 9, "activation": "tanh"},
        {"input_dim": 9, "output_dim": 6, "activation": "tanh"},
        {"input_dim": 6, "output_dim": 3, "activation": "log_softmax"},
    ]

    # Initialize the model, forward-propagate with the test tensor, print out
    # the result
    model = Net(nn_architecture)
    t = T.Tensor(np.random.uniform(-100, 100, (3, 4, 3)))
    pprint(model.forward(t).data)

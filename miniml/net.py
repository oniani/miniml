# type: ignore

import numpy as np

from layer import Layer, Linear, Sigmoid
from tensor import Tensor
from typing import List


class Net:
    def __init__(self) -> None:
        self._layers: List[Layer] = []

    def __str__(self) -> str:
        """A string representation of the network."""

        layers: List[str] = [l.__class__.__name__ for l in self._layers]

        out: List[str] = []
        for idx, layer in enumerate(layers):
            out.append(f"({idx + 1}) {layer}")

        return "\n".join(out)

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the neural architecture."""

        self._layers.append(layer)

    def predict(self, x: Tensor) -> Tensor:
        """Make a prediction."""

        for layer in self._layers:
            x = layer.forward(x)

        return x


if __name__ == "__main__":
    # Some pretty-printing
    from pprint import pprint

    # Random seed for consistent results
    np.random.seed(57)

    # Dummy data
    t = Tensor(np.random.uniform(-100, 100, (3, 4, 3)))

    # The model
    model = Net()
    model.add_layer(Linear(10, 15))
    model.add_layer(Linear(15, 20))
    model.add_layer(Sigmoid(20))

    # Print the prediction
    pprint(model.predict(t).data)
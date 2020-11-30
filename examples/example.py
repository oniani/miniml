#!/usr/bin/env python3

import pandas as pd

from miniml.tensor import Tensor
from miniml.layer import Linear
from miniml.activation import ReLU, Sigmoid
from miniml.net import Model


def generate_data(num: int) -> pd.DataFrame:
    """Learning OR"""

    data: list[list[int]] = []
    for _ in range(num):
        data.append([1, 1, 1])
        data.append([1, 0, 1])
        data.append([0, 1, 1])
        data.append([0, 0, 0])

    return pd.DataFrame(data, columns=["fst", "snd", "res"])


def main() -> None:
    """The main function."""

    # Generate data
    data = generate_data(2)

    # GPU
    x = Tensor(data[["fst", "snd"]].to_numpy(), gpu=True)
    # y = Tensor(data["res"].values, gpu=True)

    # Create the model
    model = Model()

    # Add layers
    model.add_layer(Linear(x.shape[0], 5, gpu=True))
    model.add_layer(ReLU(5))

    model.add_layer(Linear(5, 2, gpu=True))
    model.add_layer(ReLU(2))

    model.add_layer(Linear(2, 1, gpu=True))
    model.add_layer(Sigmoid(1))

    # Train the model
    # model.train(x_train=x.T, y_train=y, lr=3e-4, epochs=100)
    print(model.predict(x))


if __name__ == "__main__":
    main()

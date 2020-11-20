# type: ignore

"""
A Tensor module on top of Numpy arrays.

TODO: Implement the reverse mode autodiff to compute gradients it will have
      to go backward through the computation graph.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, List, Tuple, Union
import warnings

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


# Initialize the context
CONTEXT: cl.Context = cl.create_some_context(answers=[0, 1])

# Instantiate a queue
QUEUE: cl.CommandQueue = cl.CommandQueue(CONTEXT)


class Tensor:
    """A tensor class. Computations can be delegated to the GPU."""

    def __init__(
        self, data: Union[List, np.ndarray], gpu: bool = False
    ) -> None:
        """Initialize variables."""

        if isinstance(data, list):
            self._data: np.ndarray = np.array(data, dtype=np.float32)

        elif isinstance(data, np.ndarray):
            if data.dtype != np.dtype("float32"):
                self._data: np.ndarray = data.astype(np.float32)
            else:
                self._data: np.ndarray = data

        elif isinstance(data, cl_array.Array):
            self._data: cl_arr.Array = data
            warnings.warn("Warning: using `pyopencl.array.Array` directly")

        else:
            raise TypeError(
                f"Expected `np.ndarray` or `list`, got `{type(data)}`"
            )

        self._gpu: bool = gpu

        if self._gpu:
            self._data = cl_array.to_device(QUEUE, self._data)

    def __str__(self) -> str:
        """A string representation of the tensor."""

        return f"Tensor({self._data}, gpu={self._gpu})"

    def cpu(self) -> Tensor:
        """Load the data into CPU."""

        if self._gpu:
            self._data = self._data.get()
            self._gpu = False

        return self

    def gpu(self) -> Tensor:
        """Load the data into GPU."""

        if not self._gpu:
            self._data = cl_array.to_device(QUEUE, self._data)
            self._gpu = True

        return self

    def _setup(fun: Callable) -> Callable:
        """A decorator for checking types and handling GPU tensors."""

        @wraps(fun)
        def wrapper(self, other: Tensor) -> Callable:
            if not isinstance(other, Tensor):
                raise TypeError(f"{other} must be a `Tensor`")

            if self._gpu and not other._gpu:
                other = other.gpu()
            elif not self._gpu and other._gpu:
                self = self.gpu()

            return fun(self, other)

        return wrapper

    @_setup
    def __add__(self, other: Tensor) -> Tensor:
        """Add two tensors."""

        return Tensor(self._data + other._data)

    @_setup
    def __iadd__(self, other: Tensor) -> Tensor:
        """Add two tensors in-place."""

        self = Tensor(self._data + other._data)
        return self

    @_setup
    def __sub__(self, other: Tensor) -> Tensor:
        """Subtract two tensors."""

        return Tensor(self._data - other._data)

    @_setup
    def __isub__(self, other: Tensor) -> Tensor:
        """Subtract two tensors in-place."""

        self = Tensor(self._data - other._data)
        return self

    @_setup
    def __mul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors."""

        return Tensor(self._data * other._data)

    @_setup
    def __imul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors in-place."""

        self = Tensor(self._data * other._data)
        return self

    @_setup
    def __truediv__(self, other: Tensor) -> Tensor:
        """Divide two tensors."""

        return Tensor(self._data / other._data)

    @_setup
    def __itruediv__(self, other: Tensor) -> Tensor:
        """Divide two tensors in-place."""

        self = Tensor(self._data / other._data)
        return self

    @_setup
    def __matmul__(self, other: Tensor) -> Tensor:
        """Multiply two matrices."""

        return Tensor(self._data @ other._data)

    @_setup
    def __imatmul__(self, other: Tensor) -> Tensor:
        """Multiply two matrices in-place."""

        self._data = self._data @ other._data
        return self

    @_setup
    def __eq__(self, other: Tensor) -> Tensor:
        """Compare two tensors for equality."""

        if self._gpu:
            return (self._data.get() == other._data.get()).all()

        return (self._data == other._data).all()

    @_setup
    def __neq__(self, other: Tensor) -> Tensor:
        """Compare two tensors for inequality."""

        if self._gpu:
            return (self._data.get() == other._data.get()).all()

        return (self._data != other._data).any()

    @property
    def data(self) -> np.ndarray:
        """The data inside of the tensor."""

        return self._data

    @data.setter
    def data(self, other: Union[List, np.ndarray]) -> Tensor:
        """Set the data inside of the tensor."""

        self._data = other

    @property
    def dtype(self) -> int:
        """The data type of the tensor."""

        return np.ndim(self._data.dtype)

    @property
    def shape(self) -> Tuple:
        """The shape of the tensor."""

        return self._data.shape

    @property
    def ndim(self) -> int:
        """The dimensions of the tensor."""

        return np.ndim(self._data)

    @property
    def nbytes(self) -> int:
        """Return the number of bytes."""

        return self._data.nbytes

    @property
    def gpu_state(self) -> bool:
        """The GPU value of the tensor."""

        return self._gpu

    @staticmethod
    def dot(m1: Tensor, m2: Tensor) -> Tensor:
        """Returns a dot product of two tensors."""

        return Tensor(np.dot(m1.data, m2.data))

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
        self,
        data: Union[List, np.ndarray],
        gpu: bool = False,
        _children: List[Tensor] = [],
        _op: str = "",
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
            self._data: cl_array.Array = data
            # warnings.warn(
            #     "Warning: using `pyopencl.array.Array` directly. "
            #     "Tensors are loaded into the GPU. "
            #     "GPU is enabled."
            # )

        else:
            raise TypeError(
                f"Expected `np.ndarray` or `list`, got `{type(data)}`"
            )

        self._children: List[Tensor] = list(_children)
        self._op: str = _op
        self._gpu: bool = gpu
        self._grad: int = 0

        if self._gpu and not isinstance(self._data, cl_array.Array):
            self._data = cl_array.to_device(QUEUE, self._data)

    def __str__(self) -> str:
        """A string representation of the tensor."""

        return (
            f"Tensor(data={self._data},\n"
            f"       gpu={self._gpu},\n"
            f"       _children={list(range(len(self._children)))},\n"
            f"       _op={self._op}\n"
            ")"
        )

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

    def _typecheck(fun: Callable) -> Callable:
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

    def backward(self):
        """Perform a backward pass."""

        # Construct the reverse topological order of all of the children in the
        # computation graph
        reverse_topological_ordering = []
        visited = []

        def topological_ordering(v):
            if v not in visited:
                visited.append(v)
                for child in v._children:
                    topological_ordering(child)
                reverse_topological_ordering.append(v)

        topological_ordering(self)
        reverse_topological_ordering.reverse()

        # Gradient of a tensor with respect to itself is 1
        self.grad = 1

        # Apply the chain rule to the children
        for vertex in reverse_topological_ordering:
            vertex._grad()

    @_typecheck
    def __add__(self, other: Tensor) -> Tensor:
        """Add two tensors."""

        out = Tensor(self._data + other._data, self._gpu, [self, other], "+")

        def _grad():
            self._grad += out._grad
            other._grad += out._grad

        out._grad = _grad

        return out

    @_typecheck
    def __iadd__(self, other: Tensor) -> Tensor:
        """Add two tensors in-place."""

        self._data += other._data
        return self

    @_typecheck
    def __sub__(self, other: Tensor) -> Tensor:
        """Subtract two tensors."""

        out = Tensor(self._data - other._data, self._gpu, [self, other], "-")

        def _grad():
            self._grad -= out._grad
            other._grad -= out._grad

        out._grad = _grad

        return out

    @_typecheck
    def __isub__(self, other: Tensor) -> Tensor:
        """Subtract two tensors in-place."""

        self._data -= other._data
        return self

    @_typecheck
    def __mul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors."""

        out = Tensor(self._data * other._data, self._gpu, [self, other], "*")

        def _grad():
            self._grad += other._grad * out._grad
            other._grad += slef._grad * out._grad

        out._grad = _grad

        return out

    @_typecheck
    def __imul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors in-place."""

        self._data *= other._data
        return self

    @_typecheck
    def __truediv__(self, other: Tensor) -> Tensor:
        """Divide two tensors."""

        return Tensor(self._data / other._data)

    @_typecheck
    def __itruediv__(self, other: Tensor) -> Tensor:
        """Divide two tensors in-place."""

        self._data /= other._data
        return self

    @_typecheck
    def __matmul__(self, other: Tensor) -> Tensor:
        """Multiply two matrices."""

        return Tensor(self._data @ other._data)

    @_typecheck
    def __imatmul__(self, other: Tensor) -> Tensor:
        """Multiply two matrices in-place."""

        # self._data @= other._data
        # return self

        raise NotImplementedError(
            "In-place matrix multiplication is not (yet) supported. "
            "Use 'a = a @ b' instead of 'a @= b'."
        )

    @_typecheck
    def __eq__(self, other: Tensor) -> Tensor:
        """Compare two tensors for equality."""

        if self._gpu:
            return (self._data.get() == other._data.get()).all()

        return (self._data == other._data).all()

    @_typecheck
    def __neq__(self, other: Tensor) -> Tensor:
        """Compare two tensors for inequality."""

        if self._gpu:
            return (self._data.get() == other._data.get()).any()

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


if __name__ == "__main__":
    t1 = Tensor([1, 2, 3], gpu=True)
    t2 = Tensor([1, 2, 3], gpu=True)
    print(Tensor.dot(t1, t2))
    # print(t1)
    # print(t2)

    # t3 = t1 * t2

    # Fix TypeError: unsupported operand type(s) for *: 'int' and 'function'
    # print(t3.backward())

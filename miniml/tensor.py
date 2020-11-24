# type: ignore

"""
A Tensor module on top of Numpy arrays.

TODO: Implement the reverse mode autodiff to compute gradients. It will have
      to go backward through the computation graph.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, List, Tuple, Union

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
        self, data: Union[List, np.ndarray, cl_array.Array], gpu: bool = False,
    ) -> None:
        """Initialize variables."""

        self._gpu: bool = gpu

        if isinstance(data, list):
            self._data: np.ndarray = np.array(data, dtype=np.float32)

        elif isinstance(data, np.ndarray):
            if data.dtype != np.dtype("float32"):
                self._data: np.ndarray = data.astype(np.float32)
            else:
                self._data: np.ndarray = data

        elif isinstance(data, cl_array.Array):
            self._data: cl_array.Array = data
            self._gpu = True

        else:
            raise TypeError(
                f"Expected `np.ndarray` or `list`, got `{type(data)}`"
            )

        if self._gpu and not isinstance(self._data, cl_array.Array):
            self._data = cl_array.to_device(QUEUE, self._data)

    def __repr__(self) -> str:
        """A string representation of a tensor."""

        return (
            "Tensor(\n"
            f"    data={self._data},\n"
            f"    gpu={self._gpu}\n"
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

    @property
    def gpu_state(self) -> bool:
        """The GPU value of a tensor."""

        return self._gpu

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

    @_typecheck
    def __add__(self, other: Tensor) -> Tensor:
        """Add two tensors."""

        return Tensor(self._data + other._data, gpu=self._gpu or other._gpu)

    @_typecheck
    def __iadd__(self, other: Tensor) -> Tensor:
        """Add two tensors in-place."""

        self._data += other._data
        return self

    @_typecheck
    def __sub__(self, other: Tensor) -> Tensor:
        """Subtract two tensors."""

        return Tensor(self._data - other._data, gpu=self._gpu or other._gpu)

    @_typecheck
    def __isub__(self, other: Tensor) -> Tensor:
        """Subtract two tensors in-place."""

        self._data -= other._data
        return self

    @_typecheck
    def __mul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors."""

        return Tensor(self._data * other._data, gpu=self._gpu or other._gpu)

    @_typecheck
    def __imul__(self, other: Tensor) -> Tensor:
        """Multiply two tensors in-place."""

        self._data *= other._data
        return self

    @_typecheck
    def __truediv__(self, other: Tensor) -> Tensor:
        """Divide two tensors."""

        return Tensor(self._data / other._data, gpu=self._gpu or other._gpu)

    @_typecheck
    def __itruediv__(self, other: Tensor) -> Tensor:
        """Divide two tensors in-place."""

        self._data /= other._data
        return self

    @_typecheck
    def __matmul__(self, other: Tensor) -> Tensor:
        """Multiply two matrices."""

        return Tensor(self._data @ other._data, gpu=self._gpu or other._gpu)

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

        return (self._data == other._data).all()

    @_typecheck
    def __neq__(self, other: Tensor) -> Tensor:
        """Compare two tensors for inequality."""

        return (self._data == other._data).all()

    def view(self, dtype: np.dtype) -> None:
        """Returns view of array with the same data. If dtype is different from
           current dtype, the actual bytes of memory will be reinterpreted.
        """

        return Tensor(self._data.view(dtype), gpu=self._gpu)

    def squeeze(self) -> None:
        """Returns a view of the array with dimensions of length 1 removed."""

        return Tensor(self._data.squeeze(), gpu=self._gpu)

    @property
    def data(self) -> Union[np.ndarray, cl_array.Array]:
        """The data inside of a tensor."""

        return self._data

    @data.setter
    def data(self, other: Union[List, np.ndarray, cl_array.Array]) -> None:
        """Set the data inside of a tensor."""

        # TODO: Need the checks!
        self._data = other

    @property
    def T(self) -> Tensor:
        """Returns a transpose of a tensor."""

        return Tensor(self._data.T, gpu=self._gpu)

    @property
    def dtype(self) -> int:
        """The data type of a tensor."""

        return self._data.dtype

    def flags(self) -> Tuple:
        """Return an object with attributes `c_contiguous`, `f_contiguous` and
           `forc`, which may be used to query contiguity properties in analogy
           to `numpy.ndarray.flags`.
        """

        self._data.size

    @property
    def ndim(self) -> int:
        """The dimensions of a tensor."""

        return self._data.ndim

    @property
    def nbytes(self) -> int:
        """Return the number of bytes."""

        return self._data.nbytes

    @property
    def shape(self) -> Tuple:
        """The tuple of lengths of each dimension in the array."""

        return self._data.shape

    @property
    def strides(self) -> Tuple:
        """Tuple of bytes to step in each dimension."""

        self._data.strides

    @property
    def size(self) -> Tuple:
        """The number of meaningful entries in the array."""

        self._data.size

    @staticmethod
    @_typecheck
    def dot(m1: Tensor, m2: Tensor) -> Tensor:
        """Returns a dot product of two tensors."""

        if m1._gpu or m2._gpu:
            return Tensor(cl_array.dot(m1.data, m2.data), gpu=True)

        return Tensor(np.dot(m1.data, m2.data))

    @staticmethod
    def flatten(t: Tensor) -> Tensor:
        """Returns flattened array containing the same data."""

        return Tensor(t._data.ravel(), gpu=t._gpu)

    @staticmethod
    def reshape(t: Tensor, shape: Tuple) -> Tensor:
        """Returns an array containing the same data with a new shape."""

        if t._gpu:
            return Tensor(cl_array.reshape(t._data, shape), gpu=True)

        return Tensor(np.reshape(t._data, shape), gpu=False)

    @staticmethod
    def log(t: Tensor) -> Tensor:
        """Returns a natural logarithm of a tensor."""

        if t._gpu:
            return Tensor(cl.math.log(t._data), gpu=True)

        return Tensor(np.log(t._data))

    @staticmethod
    def exp(t: Tensor) -> Tensor:
        """Returns a natural exponent of a tensor."""

        if t._gpu:
            return Tensor(cl.math.exp(t._data), gpu=True)

        return Tensor(np.exp(t._data))

    @staticmethod
    def max(t: Tensor) -> Tensor:
        """Returns the maximum of a tensor."""

        if t._gpu:
            return Tensor(cl_array.max(t._data), gpu=True)

        return Tensor(np.max(t._data))

    @staticmethod
    def maximum(t: Tensor) -> Tensor:
        """Returns the maximum of a tensor."""

        if t._gpu:
            return Tensor(cl_array.max(t._data), gpu=True)

        return Tensor(np.maximum(t._data))

    @staticmethod
    def min(t: Tensor) -> Tensor:
        """Returns the minimum of a tensor."""

        if t._gpu:
            return Tensor(cl_array.max(t._data), gpu=True)

        return Tensor(np.min(t._data))

    @staticmethod
    def minimum(t: Tensor, val: float) -> Tensor:
        """Returns the minimum of a tensor."""

        if t._gpu:
            return Tensor(cl_array.max(t._data), gpu=True)

        return Tensor(np.minimum(t._data, val))

    @staticmethod
    def normal(shape: Tuple = (1, 1), gpu=False) -> Tensor:
        """Draw random samples from a normal (Gaussian) distribution."""

        if gpu:
            return Tensor(
                cl.clrandom.PhiloxGenerator(CONTEXT).normal(
                    cq=QUEUE, shape=shape, dtype=np.float32
                ),
                gpu=True,
            )

        return Tensor(np.random.normal(size=shape).astype(np.float32))

    @staticmethod
    def power(t: Tensor, exponent: Union[float, int, Tensor]) -> Tensor:
        """Draw random samples from a normal (Gaussian) distribution."""

        if isinstance(exponent, Tensor):
            return Tensor(t._data ** exponent._data, gpu=t._gpu)

        return Tensor(t._data ** exponent, gpu=t._gpu)

    @staticmethod
    def rand(shape: Tuple = (1, 1), gpu=False) -> Tensor:
        """Returns a tensor of random values in a given shape."""

        if gpu:
            return Tensor(cl.clrandom.rand(QUEUE, shape, np.float32), gpu=True)

        return Tensor(np.random.rand(*shape).astype(np.float32))

    @staticmethod
    def sum(t: Tensor) -> Tensor:
        """Returns a sum of a tensor."""

        if t._gpu:
            return Tensor(cl_array.sum(t._data), gpu=True)

        return Tensor(np.sum(t._data))

    @staticmethod
    def uniform(
        min: float = 0.0, max: float = 1.0, shape: Tuple = (1, 1), gpu=False
    ) -> Tensor:
        """Draw samples from a uniform distribution."""

        if gpu:
            return Tensor(
                cl.clrandom.PhiloxGenerator(CONTEXT).uniform(
                    cq=QUEUE, shape=shape, dtype=np.float32, a=min, b=max
                ),
                gpu=True,
            )

        return Tensor(
            np.random.uniform(min, max, size=shape).astype(np.float32)
        )

    @staticmethod
    def zeros(shape: Tuple = (1, 1), gpu=False) -> Tensor:
        """Return a new tensor of given shape and type, filled with zeros."""

        if gpu:
            return Tensor(cl_array.zeros(QUEUE, shape, np.float32), gpu=True)

        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def zeros_like(t: Tensor, gpu=False) -> Tensor:
        """Return a tensor of zeros with the same shape and type as a given
           tensor.
        """

        if gpu:
            return Tensor(cl_array.zeros_like(arr), gpu=True)

        return Tensor(np.zeros_like(t._data, dtype=np.float32))

    @staticmethod
    def zeros(shape: Tuple = (1, 1), gpu=False) -> Tensor:
        """Return a new tensor of given shape and type, filled with zeros."""

        if gpu:
            return Tensor(cl_array.zeros(QUEUE, shape, np.float32), gpu=True)

        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def zeros_like(t: Tensor, gpu=False) -> Tensor:
        """Return a tensor of zeros with the same shape and type as a given
           tensor.
        """

        if gpu:
            return Tensor(cl_array.zeros_like(arr), gpu=True)

        return Tensor(np.zeros_like(t._data, dtype=np.float32))

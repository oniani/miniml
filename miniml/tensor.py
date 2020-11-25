# type: ignore

"""
A Tensor module on top of Numpy arrays.

TODO: Implement the reverse mode autodiff to compute gradients. It will have
      to go backward through the computation graph.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clmath as cl_math


# Initialize the context
CONTEXT: cl.Context = cl.create_some_context(answers=[0, 1])

# Instantiate a queue
QUEUE: cl.CommandQueue = cl.CommandQueue(CONTEXT)


class Tensor:
    """A tensor class. Computations can be delegated to the GPU."""

    # Types to be used inside the class
    Data = Union[List, np.ndarray, cl_array.Array]
    Scalar = Union[float, int]
    UTS = Union["Tensor", Scalar]

    def __init__(self, data: Data, gpu: bool = False) -> None:
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
        """A representation of a tensor."""

        return f"Tensor(\n    data={self._data},\n    gpu={self._gpu}\n)"

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

    def __add__(self, other: UTS) -> Tensor:
        """Add two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data + other, gpu=self._gpu)

        return Tensor(self._data + other._data, gpu=self._gpu or other._gpu)

    __radd__ = __add__

    def __iadd__(self, other: UTS) -> Tensor:
        """Add two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data += other
        else:
            self._data += other._data

        return self

    def __sub__(self, other: Union[Tenspr, float, int]) -> Tensor:
        """Subtract two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data - other, gpu=self._gpu)

        return Tensor(self._data - other._data, gpu=self._gpu or other._gpu)

    __rsub__ = __sub__

    def __isub__(self, other: UTS) -> Tensor:
        """Subtract two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data -= other
        else:
            self._data -= other._data

        return self

    def __mul__(self, other: UTS) -> Tensor:
        """Multiply two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data * other, gpu=self._gpu)

        return Tensor(self._data * other._data, gpu=self._gpu or other._gpu)

    __rmul__ = __mul__

    def __imul__(self, other: UTS) -> Tensor:
        """Multiply two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data *= other
        else:
            self._data *= other._data

        return self

    def __truediv__(self, other: UTS) -> Tensor:
        """Divide two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data / other, gpu=self._gpu)

        return Tensor(self._data / other._data, gpu=self._gpu or other._gpu)

    __rtruediv__ = __truediv__

    def __itruediv__(self, other: UTS) -> Tensor:
        """Divide two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data /= other
        else:
            self._data /= other._data

        return self

    def __lt__(self, other: UTS) -> Tensor:
        """Compare two tensors for less than."""

        if not isinstance(other, Tensor):
            return Tensor(self._data < other, gpu=self._gpu)

        return Tensor(self._data < other._data, gpu=self._gpu or other._gpu)

    def __le__(self, other: UTS) -> Tensor:
        """Compare two tensors for less than or equal."""

        if not isinstance(other, Tensor):
            return Tensor(self._data <= other, gpu=self._gpu)

        return Tensor(self._data <= other._data, gpu=self._gpu or other._gpu)

    def __eq__(self, other: UTS) -> Tensor:
        """Compare two tensors for equality."""

        if not isinstance(other, Tensor):
            return Tensor(self._data == other, gpu=self._gpu)

        return Tensor(self._data == other._data, gpu=self._gpu or other._gpu)

    def __ne__(self, other: UTS) -> Tensor:
        """Compare two tensors for inequality."""

        if not isinstance(other, Tensor):
            return Tensor(self._data != other, gpu=self._gpu)

        return Tensor(self._data != other._data, gpu=self._gpu or other._gpu)

    def __ge__(self, other: UTS) -> Tensor:
        """Compare two tensors for greater than."""

        if not isinstance(other, Tensor):
            return Tensor(self._data >= other, gpu=self._gpu)

        return Tensor(self._data >= other._data, gpu=self._gpu or other._gpu)

    def __gt__(self, other: UTS) -> Tensor:
        """Compare two tensors for greater than."""

        if not isinstance(other, Tensor):
            return Tensor(self._data > other, gpu=self._gpu)

        return Tensor(self._data > other._data, gpu=self._gpu or other._gpu)

    def __ng__(self) -> Tensor:
        """Return the negated Tensor."""

        return Tensor(-self._data)

    def all(self) -> bool:
        """True if all values of the Tensor are `True`, False otherwise."""

        return self._data.all()

    def any(self) -> bool:
        """True if at least one value of the Tensor is `True`, False otherwise
        """

        return self._data.any()

    def view(self, dtype: np.dtype) -> None:
        """Returns view of array with the same data. If dtype is different from
           current dtype, the actual bytes of memory will be reinterpreted.
        """

        return Tensor(self._data.view(dtype), gpu=self._gpu)

    def astype(self, dtype: np.dtype) -> Tensoor:
        """Return a copy of self, cast to dtype."""

        return Tensor(self._data.astype(dtype), gpu=self._gpu)

    def squeeze(self) -> None:
        """Returns a view of the array with dimensions of length 1 removed."""

        return Tensor(self._data.squeeze(), gpu=self._gpu)

    @property
    def data(self) -> Union[np.ndarray, cl_array.Array]:
        """The data inside of a tensor."""

        return self._data

    @data.setter
    def data(self, new_data: Union[List, np.ndarray, cl_array.Array]) -> None:
        """Set the data inside of a tensor."""

        if isinstance(new_data, list):
            self._data: np.ndarray = np.array(new_data, dtype=np.float32)
            self._gpu: bool = False

        elif isinstance(data, np.ndarray):
            if data.dtype != np.dtype("float32"):
                self._data: np.ndarray = data.astype(np.float32)
            else:
                self._data: np.ndarray = data

            self._gpu: bool = False

        elif isinstance(data, cl_array.Array):
            self._data: cl_array.Array = data
            self._gpu = True

        else:
            raise TypeError(
                f"Expected `np.ndarray` or `list`, got `{type(data)}`"
            )

    @property
    def T(self) -> Tensor:
        """Returns a transpose of a tensor."""

        return Tensor(self._data.T, gpu=self._gpu)

    @property
    def dtype(self) -> int:
        """The data type of a tensor."""

        return self._data.dtype

    @property
    def flags(self) -> Tuple:
        """Return an object with attributes `c_contiguous`, `f_contiguous` and
           `forc`, which may be used to query contiguity properties in analogy
           to `numpy.ndarray.flags`.
        """

        return self._data.size

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


class Ops:
    """Tensor operations."""

    @staticmethod
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
    def fill(shape: Tuple, val: Union[float, int], gpu=False) -> Tensor:
        """Fill the array with scalar."""

        if gpu:
            return Tensor(
                cl_array.empty(QUEUE, shape, dtype=np.float32).fill(val),
                gpu=True,
            )

        return Tensor(np.full(shape, val))

    @staticmethod
    def where(cond: Tensor, fst: UTS, snd: UTS) -> Tensor:
        """Fill the array with scalar."""

        if cond._gpu:
            shape = cond._data.shape
            if not isinstance(fst, Tensor) and isinstance(snd, Tensor):
                snd = snd._data
                fst = cl_array.empty(QUEUE, shape, dtype=np.float32).fill(fst)

            elif isinstance(fst, Tensor) and not isinstance(snd, Tensor):
                fst = fst._data
                snd = cl_array.empty(QUEUE, shape, dtype=np.float32).fill(snd)

            elif not isinstance(fst, Tensor) and not isinstance(snd, Tensor):
                fst = cl_array.empty(QUEUE, shape, dtype=np.float32).fill(fst)
                snd = cl_array.empty(QUEUE, shape, dtype=np.float32).fill(snd)

            return Tensor(cl_array.if_positive(cond._data, fst, snd), gpu=True)

        if not isinstance(fst, Tensor) and isinstance(snd, Tensor):
            return Tensor(np.where(cond._data, fst, snd._data))

        if isinstance(fst, Tensor) and not isinstance(snd, Tensor):
            return Tensor(np.where(cond._data, fst._data, snd))

        if not isinstance(fst, Tensor) and not isinstance(snd, Tensor):
            return Tensor(np.where(cond._data, fst, snd))

        return Tensor(np.where(cond._data, fst._data, snd._data))

    @staticmethod
    def reshape(t: Tensor, shape: Tuple) -> Tensor:
        """Returns an array containing the same data with a new shape."""

        if t._gpu:
            return Tensor(cl_array.reshape(t._data, shape), gpu=True)

        return Tensor(np.reshape(t._data, shape))

    @staticmethod
    def log(t: Tensor) -> Tensor:
        """Returns a natural logarithm of a tensor."""

        if t._gpu:
            return Tensor(cl_math.log(t._data), gpu=True)

        return Tensor(np.log(t._data))

    @staticmethod
    def tanh(t: Tensor) -> Tensor:
        """Returns a tanh of a tensor."""

        if t._gpu:
            return Tensor(cl_math.tanh(t._data), gpu=True)

        return Tensor(np.tanh(t._data))

    @staticmethod
    def exp(t: Tensor) -> Tensor:
        """Returns a natural exponent of a tensor."""

        if t._gpu:
            return Tensor(cl_math.exp(t._data), gpu=True)

        return Tensor(np.exp(t._data))

    @staticmethod
    def maximum(t: Tensor, uts: UTS) -> Tensor:
        """Returns the maximum of a tensor."""

        if t._gpu:
            if not isinstance(uts, Tensor):
                return Tensor(
                    cl_array.maximum(
                        t._data,
                        cl_array.empty(QUEUE, t.shape, dtype=np.float32).fill(
                            uts
                        ),
                    ),
                    gpu=t._gpu,
                )

            return Tensor(cl_array.maximum(t._data, uts._data), gpu=t._gpu)

        if not isinstance(uts, Tensor):
            return Tensor(np.maximum(t._data, uts))

        return Tensor(np.maximum(t._data, uts._data))

    @staticmethod
    def minimum(t: Tensor, uts: UTS) -> Tensor:
        """Returns the minimum of a tensor."""

        if t._gpu:
            if not isinstance(uts, Tensor):
                return Tensor(
                    cl_array.minimum(
                        t._data,
                        cl_array.empty(QUEUE, t.shape, dtype=np.float32).fill(
                            uts
                        ),
                    ),
                    gpu=t._gpu,
                )

            return Tensor(cl_array.minimum(t._data, uts._data), gpu=t._gpu)

        if not isinstance(uts, Tensor):
            return Tensor(np.minimum(t._data, uts))

        return Tensor(np.minimum(t._data, uts._data))

    @staticmethod
    def power(t: Tensor, exponent: Union[float, int, Tensor]) -> Tensor:
        """Draw random samples from a normal (Gaussian) distribution."""

        if not isinstance(exponent, Tensor):
            return Tensor(t._data ** exponent, gpu=t._gpu)

        return Tensor(t._data ** exponent._data, gpu=t._gpu or exponent._gpu)

    @staticmethod
    def square(t: Tensor) -> Tensor:
        """Return a square-valued tensor."""

        return Tensor(t._data ** 2, gpu=t._gpu)

    @staticmethod
    def transpose(t: Tensor) -> Tensor:
        """Returns a transpose of a tensor."""

        if t._gpu:
            return Tensor(cl_array.transpose(t._data), gpu=True)

        return Tensor(np.transpose(t._data), gpu=t._gpu)

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
            return Tensor(cl_array.zeros_like(t._data), gpu=True)

        return Tensor(np.zeros_like(t._data, dtype=np.float32))


class Random:
    """Random number generation for tensors."""

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
    def rand(shape: Tuple = (1, 1), gpu=False) -> Tensor:
        """Returns a tensor of random values in a given shape."""

        if gpu:
            return Tensor(cl.clrandom.rand(QUEUE, shape, np.float32), gpu=True)

        return Tensor(np.random.rand(*shape).astype(np.float32))

    @staticmethod
    def uniform(
        shape: Tuple = (1, 1), min: float = 0.0, max: float = 1.0, gpu=False
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


class Reduce:
    """Reduction operations on tensors."""

    @staticmethod
    def max(t: Tensor) -> float:
        """Returns the maximum of a tensor."""

        if t._gpu:
            return cl_array.max(t._data).get().flat[0]

        return np.max(t._data)

    @staticmethod
    def min(t: Tensor) -> Tensor:
        """Returns the minimum of a tensor."""

        if t._gpu:
            return cl_array.max(t._data).get().flat[0]

        return Tensor(np.min(t._data))

    @staticmethod
    def sum(t: Tensor) -> Scalar:
        """Returns a sum of a tensor."""

        if t._gpu:
            return cl_array.sum(t._data).get().flat[0]

        return np.sum(t._data)

    @staticmethod
    def mean(t: Tensor) -> float:
        """Returns the mean of a tensor."""

        if t._gpu:
            return cl_array.sum(t._data).get().flat[0] / t._data.size

        return np.mean(t._data)

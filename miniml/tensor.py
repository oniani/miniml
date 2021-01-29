# type: ignore
"""
A Tensor module on top of Numpy arrays.

TODO: Implement the reverse mode autodiff to compute gradients. It will have
      to go backward through the computation graph.
"""

from __future__ import annotations
from typing import Union

import os

import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.clmath as clmath
import pyopencl.clrandom as clrandom
import pyopencl.bitonic_sort as clbitonicsort


# Initialize the context
CONTEXT: cl.Context = cl.create_some_context(answers=[0, 1])

# Instantiate a queue
QUEUE: cl.CommandQueue = cl.CommandQueue(CONTEXT)

# OpenCL options
CLOPTS: str = "-cl-mad-enable -cl-fast-relaxed-math"

# Scalar type
Scalar = Union[float, int, np.float32]


def readcl(filename: str) -> str:
    """Read an OpenCL file and return it as a string."""

    dirname: str = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    with open(os.path.join(dirname, "opencl", filename)) as file:
        data: str = file.read()

    return data


class Tensor:
    """A tensor class. Computations can be delegated to the GPU."""

    def __init__(
        self, data: Union[cl.array.Array, list, np.ndarray], gpu: bool = False
    ) -> None:
        """Initialize variables."""

        self._gpu: bool = gpu

        if isinstance(data, list):
            self._data: np.ndarray = np.array(data, dtype=np.float32)

            if self._gpu:
                self._data = clarray.to_device(QUEUE, self._data)

        elif isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                # NOTE: The NumPy array has to be converted into a list first.
                #       Otherwise, the operations on cpu and gpu produce
                #       different results. This behavior can be caused by many
                #       reasons including OpenCL and even the operating system
                #       itself. Some research is needed to figure out cause and
                #       eliminate extra work for rebuilding the array.
                self._data: np.ndarray = np.array(data.tolist(), np.float32)
            else:
                self._data: np.ndarray = data

            if self._gpu:
                self._data = clarray.to_device(QUEUE, self._data)

        elif isinstance(data, cl.array.Array):
            self._data: cl.array.Array = data
            self._gpu: bool = True

        else:
            raise TypeError(
                "Expected `list`, `np.ndarray`, or `pyopencl.array.Array` got "
                f"`{type(data)}`"
            )

    @property
    def data(self) -> Union[np.ndarray, cl.array.Array]:
        """The data inside of a tensor."""

        return self._data

    @data.setter
    def data(self, data: Union[cl.array.Array, list, np.ndarray]) -> None:
        """Set the data inside of a tensor."""

        if isinstance(data, list):
            self._data: np.ndarray = np.array(data, dtype=np.float32)

            if self._gpu:
                self._data = clarray.to_device(QUEUE, self._data)

        elif isinstance(data, np.ndarray):
            if data.dtype != np.dtype("float32"):
                self._data: np.ndarray = data.astype(np.float32)
            else:
                self._data: np.ndarray = data

            if self._gpu:
                self._data = clarray.to_device(QUEUE, self._data)

        elif isinstance(data, cl.array.Array):
            self._data: cl.array.Array = data
            self._gpu: bool = True

        else:
            raise TypeError(
                "Expected `list`, `np.ndarray`, or `pyopencl.array.Array` got "
                f"`{type(data)}`"
            )

    def to_cpu(self) -> Tensor:
        """Load the data into CPU."""

        if self._gpu:
            self._data = self._data.get()
            self._gpu = False

        return self

    def to_gpu(self) -> Tensor:
        """Load the data into GPU."""

        if not self._gpu:
            self._data = clarray.to_device(QUEUE, self._data)
            self._gpu = True

        return self

    def to_numpy(self) -> np.ndarray:
        """Return a numpy ndarray."""

        if self._gpu:
            return self._data.get()

        return self._data

    @property
    def gpu(self) -> bool:
        """Return the state of the GPU."""

        return self._gpu

    def __repr__(self) -> str:
        """A representation of a tensor."""

        state: str = "GPU" if self._gpu else "CPU"
        return f"{self._data}\n\nTensor[{state}]"

    def __iter__(self) -> Union[np.ndarray, cl.array.Array]:
        """An iterator for tensors."""

        for i in self._data:
            yield i

    def __len__(self) -> int:
        """Return a length of tensors."""

        return len(self._data)

    def __getitem__(self, idx: int) -> Union[np.ndarray, cl.array.Array]:
        """Return a length of tensors."""

        return self._data[idx]

    def __setitem__(
        self, idx: int, item: Union[np.ndarray, cl.array.Array]
    ) -> None:
        """Return a length of tensors."""

        self._data[idx] = item

    def __add__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Add two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data + other, gpu=self._gpu)

        return Tensor(self._data + other._data, gpu=self._gpu or other._gpu)

    __radd__ = __add__

    def __iadd__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Add two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data += other
        else:
            self._data += other._data

        return self

    def __sub__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Subtract two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data - other, gpu=self._gpu)

        return Tensor(self._data - other._data, gpu=self._gpu or other._gpu)

    __rsub__ = __sub__

    def __isub__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Subtract two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data -= other
        else:
            self._data -= other._data

        return self

    def __mul__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Multiply two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data * other, gpu=self._gpu)

        return Tensor(self._data * other._data, gpu=self._gpu or other._gpu)

    __rmul__ = __mul__

    def __imul__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Multiply two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data *= other
        else:
            self._data *= other._data

        return self

    def __truediv__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Divide two tensors."""

        if not isinstance(other, Tensor):
            return Tensor(self._data / other, gpu=self._gpu)

        return Tensor(self._data / other._data, gpu=self._gpu or other._gpu)

    __rtruediv__ = __truediv__

    def __itruediv__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Divide two tensors in-place."""

        if not isinstance(other, Tensor):
            self._data /= other
        else:
            self._data /= other._data

        return self

    def __lt__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Less than operation for a tensor and a tensor/scalar."""

        if not isinstance(other, Tensor):
            return Tensor(self._data < other, gpu=self._gpu)

        return Tensor(self._data < other._data, gpu=self._gpu or other._gpu)

    def __le__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Less than or equal operation for a tensor and a tensor/scalar."""

        if not isinstance(other, Tensor):
            return Tensor(self._data <= other, gpu=self._gpu)

        return Tensor(self._data <= other._data, gpu=self._gpu or other._gpu)

    def __eq__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Equal to operation for a tensor and a tensor/scalar."""

        if not isinstance(other, Tensor):
            return Tensor(self._data == other, gpu=self._gpu)

        return Tensor(self._data == other._data, gpu=self._gpu or other._gpu)

    def __ne__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Not equal to operation for a tensor and a tensor/scalar."""

        if not isinstance(other, Tensor):
            return Tensor(self._data != other, gpu=self._gpu)

        return Tensor(self._data != other._data, gpu=self._gpu or other._gpu)

    def __ge__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Greater than or equal operation for a tensor and a tensor/scalar."""

        if not isinstance(other, Tensor):
            return Tensor(self._data >= other, gpu=self._gpu)

        return Tensor(self._data >= other._data, gpu=self._gpu or other._gpu)

    def __gt__(self, other: Union[Tensor, Scalar]) -> Tensor:
        """Greater than operation for a tensor and a tensor/scalar."""

        if not isinstance(other, Tensor):
            return Tensor(self._data > other, gpu=self._gpu)

        return Tensor(self._data > other._data, gpu=self._gpu or other._gpu)

    def __neg__(self) -> Tensor:
        """Return a negated tensor."""

        return Tensor(-self._data, gpu=self._gpu)

    def all(self) -> bool:
        """Returns the true value if all values of a tensor are true."""

        return self._data.all()

    def any(self) -> bool:
        """Returns the true value if at least one value of a tensor is true."""

        return self._data.any()

    def view(self, dtype: np.dtype) -> None:
        """Returns the view of a tensor with the same data. If dtype is
           different from current dtype, the actual bytes of memory will be
           reinterpreted.
        """

        return Tensor(self._data.view(dtype), gpu=self._gpu)

    def astype(self, dtype: np.dtype) -> Tensoor:
        """Return a copy of self, cast to dtype."""

        return Tensor(self._data.astype(dtype), gpu=self._gpu)

    def squeeze(self) -> None:
        """Returns a view of the tensor with dimensions of length 1 removed."""

        return Tensor(self._data.squeeze(), gpu=self._gpu)

    def sort(self) -> None:
        """Sorts a tensor, uses the parallel bitonic sort when on GPU."""

        if self._gpu:
            sorter = clbitonicsort.BitonicSort(CONTEXT)
            sorter(self._data)
        else:
            self._data.sort()

    @property
    def T(self) -> Tensor:
        """Returns a transpose of a tensor."""

        return Tensor(self._data.T, gpu=self._gpu)

    @property
    def dtype(self) -> np.dtype:
        """The data type of a tensor."""

        return self._data.dtype

    @property
    def flags(self) -> Union[cl.compyte.array.ArrayFlags, np.flagsobj]:
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
    def shape(self) -> tuple[int, ...]:
        """The tuple of lengths of each dimension in the tensor."""

        return self._data.shape

    @property
    def strides(self) -> tuple[int, ...]:
        """tuple of bytes to step in each dimension."""

        self._data.strides

    @property
    def size(self) -> int:
        """The number of meaningful entries in the tensor."""

        self._data.size


class Ops:
    """Tensor operations."""

    @staticmethod
    def dot(t1: Tensor, t2: Tensor, gpu=False) -> Tensor:
        """Returns a dot product (matrix multiplication) of two tensors."""

        if gpu:
            # Convert back to numpy ndarrays
            t1 = t1.data.get().astype(np.float32)
            t2 = t2.data.get().astype(np.float32)

            t1_w = np.int32(t1.shape[1])
            t1_h = np.int32(t1.shape[0])

            t2_w = np.int32(t2.shape[1])
            t2_h = np.int32(t2.shape[0])

            rt_h = t1_h
            rt_w = t2_w

            rt = np.empty((rt_h, rt_w)).astype(np.float32)

            # Mem flags
            mf = cl.mem_flags

            # Buffer variables
            t1_buf = cl.Buffer(
                CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t1
            )
            t2_buf = cl.Buffer(
                CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t2
            )
            rt_buf = cl.Buffer(CONTEXT, mf.WRITE_ONLY, size=rt.nbytes)

            # OpenCL program for computing a matrix multiply
            prg = cl.Program(CONTEXT, readcl("matmul.cl")).build(
                options=CLOPTS
            )

            # Perform the matrix multiplication and return the resulting tensor
            prg.matmul(
                QUEUE, rt.shape, None, t1_buf, t2_buf, rt_buf, t1_h, t2_w, t1_w
            )
            cl.enqueue_copy(QUEUE, rt, rt_buf)
            return Tensor(rt, gpu=True)

        return Tensor(np.dot(t1.data, t2.data))

    @staticmethod
    def vdot(m1: Tensor, m2: Tensor) -> Tensor:
        """Returns a dot product of two tensors."""

        if m1.gpu or m2.gpu:
            return Tensor(clarray.dot(m1.data, m2.data), gpu=True)

        return Tensor(np.vdot(m1.data, m2.data))

    @staticmethod
    def flatten(t: Tensor) -> Tensor:
        """Returns flattened tensor containing the same data."""

        return Tensor(t._data.ravel(), gpu=t.gpu)

    @staticmethod
    def fill(shape: tuple[int, ...], val: np.float32, gpu=False) -> Tensor:
        """Fill the tensor with scalar."""

        if gpu:
            return Tensor(
                clarray.empty(QUEUE, shape, dtype=np.float32).fill(val),
                gpu=True,
            )

        return Tensor(np.full(shape, val))

    @staticmethod
    def where(
        cond: Tensor, fst: Union[Tensor, Scalar], snd: Union[Tensor, Scalar],
    ) -> Tensor:
        """Fill the tensor based on a condition."""

        if cond.gpu:
            if isinstance(fst, Tensor) and isinstance(snd, Tensor):
                return Tensor(
                    clarray.if_positive(cond._data, fst._data, snd._data),
                    gpu=True,
                )

            shape: tuple[int, ...] = cond._data.shape
            if not isinstance(fst, Tensor) and isinstance(snd, Tensor):
                snd = snd._data
                fst = clarray.empty(QUEUE, shape, dtype=np.float32).fill(fst)

            elif isinstance(fst, Tensor) and not isinstance(snd, Tensor):
                fst = fst._data
                snd = clarray.empty(QUEUE, shape, dtype=np.float32).fill(snd)

            elif not isinstance(fst, Tensor) and not isinstance(snd, Tensor):
                fst = clarray.empty(QUEUE, shape, dtype=np.float32).fill(fst)
                snd = clarray.empty(QUEUE, shape, dtype=np.float32).fill(snd)

            return Tensor(clarray.if_positive(cond._data, fst, snd), gpu=True)

        if not isinstance(fst, Tensor) and isinstance(snd, Tensor):
            return Tensor(np.where(cond._data, fst, snd._data))

        if isinstance(fst, Tensor) and not isinstance(snd, Tensor):
            return Tensor(np.where(cond._data, fst._data, snd))

        if not isinstance(fst, Tensor) and not isinstance(snd, Tensor):
            return Tensor(np.where(cond._data, fst, snd))

        return Tensor(np.where(cond._data, fst._data, snd._data))

    @staticmethod
    def reshape(t: Tensor, shape: tuple) -> Tensor:
        """Returns a tensor containing the same data with a new shape."""

        if t.gpu:
            return Tensor(clarray.reshape(t._data, shape), gpu=True)

        return Tensor(np.reshape(t._data, shape))

    @staticmethod
    def log(t: Tensor) -> Tensor:
        """Returns a natural logarithm of a tensor."""

        if t.gpu:
            return Tensor(clmath.log(t._data), gpu=True)

        return Tensor(np.log(t._data))

    @staticmethod
    def tanh(t: Tensor) -> Tensor:
        """Returns a tanh of a tensor."""

        if t.gpu:
            return Tensor(clmath.tanh(t._data), gpu=True)

        return Tensor(np.tanh(t._data))

    @staticmethod
    def exp(t: Tensor) -> Tensor:
        """Returns a natural exponent of a tensor."""

        if t.gpu:
            return Tensor(clmath.exp(t._data), gpu=True)

        return Tensor(np.exp(t._data))

    @staticmethod
    def maximum(t: Tensor, uts: Union[Tensor, Scalar]) -> Tensor:
        """Returns the maximum of a tensor."""

        if t.gpu:
            if not isinstance(uts, Tensor):
                ot: cl.array.Array = clarray.empty(
                    QUEUE, t.shape, dtype=np.float32
                ).fill(uts)
                return Tensor(clarray.maximum(t._data, ot), gpu=True)

            return Tensor(clarray.maximum(t._data, uts._data), gpu=True)

        if not isinstance(uts, Tensor):
            return Tensor(np.maximum(t._data, uts))

        return Tensor(np.maximum(t._data, uts._data))

    @staticmethod
    def minimum(t: Tensor, uts: Union[Tensor, Scalar]) -> Tensor:
        """Returns the minimum of a tensor."""

        if t.gpu:
            if not isinstance(uts, Tensor):
                ot: cl.array.Array = clarray.empty(
                    QUEUE, t.shape, dtype=np.float32
                ).fill(uts)
                return Tensor(clarray.minimum(t._data, ot), gpu=True)

            return Tensor(clarray.minimum(t._data, uts._data), gpu=True)

        if not isinstance(uts, Tensor):
            return Tensor(np.minimum(t._data, uts))

        return Tensor(np.minimum(t._data, uts._data))

    @staticmethod
    def power(t: Tensor, exponent: Union[Tensor, Scalar]) -> Tensor:
        """Raise all elements of the tensor to the specified power."""

        if not isinstance(exponent, Tensor):
            return Tensor(t._data ** exponent, gpu=t.gpu)

        return Tensor(t._data ** exponent._data, gpu=t.gpu or exponent.gpu)

    @staticmethod
    def square(t: Tensor) -> Tensor:
        """Return a square-valued tensor."""

        return Tensor(t._data ** 2, gpu=t.gpu)

    @staticmethod
    def transpose(t: Tensor) -> Tensor:
        """Returns a transpose of a tensor."""

        if t.gpu:
            return Tensor(clarray.transpose(t._data), gpu=True)

        return Tensor(np.transpose(t._data), gpu=t.gpu)

    @staticmethod
    def zeros(shape: tuple = (1, 1), gpu=False) -> Tensor:
        """Return a new tensor of given shape and type, filled with zeros."""

        if gpu:
            return Tensor(clarray.zeros(QUEUE, shape, np.float32), gpu=True)

        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def zeros_like(t: Tensor, gpu=False) -> Tensor:
        """Return a tensor of zeros with the same shape and type as a given
           tensor.
        """

        if gpu:
            return Tensor(clarray.zeros_like(t._data), gpu=True)

        return Tensor(np.zeros_like(t._data, dtype=np.float32))


class Random:
    """Random number generation for tensors."""

    @staticmethod
    def normal(
        shape: Union[tuple[int, ...], int] = (1, 1), gpu=False
    ) -> Tensor:
        """Draw random samples from a normal (Gaussian) distribution."""

        if gpu:
            return Tensor(
                clrandom.PhiloxGenerator(CONTEXT).normal(
                    cq=QUEUE, shape=shape, dtype=np.float32
                ),
                gpu=True,
            )

        return Tensor(np.random.normal(size=shape).astype(np.float32))

    @staticmethod
    def rand(shape: Union[tuple[int, ...], int] = (1, 1), gpu=False) -> Tensor:
        """Returns a tensor of random values in a given shape."""

        if gpu:
            return Tensor(clrandom.rand(QUEUE, shape, np.float32), gpu=True)

        if isinstance(shape, tuple):
            return Tensor(np.random.rand(*shape).astype(np.float32))

        return Tensor(np.random.rand(shape).astype(np.float32))

    @staticmethod
    def uniform(
        shape: Union[tuple[int, ...], int] = (1, 1),
        min: float = 0.0,
        max: float = 1.0,
        gpu=False,
    ) -> Tensor:
        """Draw samples from a uniform distribution."""

        if gpu:
            return Tensor(
                clrandom.PhiloxGenerator(CONTEXT).uniform(
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
    def max(t: Tensor) -> np.float32:
        """The maximum of the values in a tensor."""

        if t.gpu:
            return clarray.max(t._data).get().flat[0]

        return np.max(t._data)

    @staticmethod
    def min(t: Tensor) -> np.float32:
        """The minimum of the values in a tensor."""

        if t.gpu:
            return clarray.min(t._data).get().flat[0]

        return np.min(t._data)

    @staticmethod
    def sum(t: Tensor) -> np.float32:
        """The sum of the values in a tensor."""

        if t.gpu:
            return clarray.sum(t._data).get().flat[0]

        return np.sum(t._data)

    @staticmethod
    def mean(t: Tensor) -> np.float32:
        """The mean of the values in a tensor."""

        if t.gpu:
            return clarray.sum(t._data).get().flat[0] / t._data.size

        return np.mean(t._data)

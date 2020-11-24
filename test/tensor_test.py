# type: ignore

import time
import pytest

import numpy as np

from miniml.tensor import Tensor as T

# Needed for testing
import pyopencl.clrandom

# Random seed for consistent results
np.random.seed(0)


class TestTensor:
    """Testing the `Tensor` class."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_class(self):
        """Initialize the tensors."""

        self.t1 = T([1, 1, 1])
        self.t2 = T([2, 2, 2])
        self.t3 = T([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.t4 = T([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])

        self.t5 = T(np.random.rand(150, 150))
        self.t6 = T(np.random.rand(150, 150))
        self.t7 = T(np.random.rand(150, 150))
        self._shape = (10_000, 5_000)

    def test_tensor_ops(self):
        """Testing `T` class operations on CPU and GPU."""

        # `+`, `-`, `*`, `/`, and `!=` operations
        assert (self.t1 + self.t2 != T([3.01, 3, 3])).any()
        assert (self.t1 - self.t2 != T([-1, -0.9999, -1])).any()
        assert (self.t1 * self.t2 != T([2, 2, 1.9999999])).any()
        assert (self.t1 / self.t2 != T([0.55, 0.5, 0.5])).any()

        # `+`, `-`, `*`, `/`, and `==` operations
        assert (self.t1 + self.t2 == T([3, 3, 3])).all()
        assert (self.t1 - self.t2 == T([-1, -1, -1])).all()
        assert (self.t1 * self.t2 == T([2, 2, 2])).all()
        assert (self.t1 / self.t2 == T([0.5, 0.5, 0.5])).all()

        # `==` and dot`
        assert (
            T.dot(self.t3, self.t4)
            == T(
                [
                    [[22, 28], [58, 64]],
                    [[49, 64], [139, 154]],
                    [[76, 100], [220, 244]],
                ]
            )
        ).all()

    def test_tensor_ops_gpu(self):
        """Testing `T` class on GPU and verifying that it is faster."""

        # Number of iterations
        its: int = 1_000

        # CPU
        cur = time.time()
        res_cpu = T([0])
        for _ in range(its):
            res_cpu = res_cpu + self.t7 * self.t5 * self.t6 * self.t7
            res_cpu = res_cpu + self.t7 * self.t5 - self.t6 * self.t7
        cpu_time = time.time() - cur

        # GPU
        cur = time.time()
        res_gpu = T([0])
        for _ in range(its):
            res_gpu = res_gpu + self.t7 * self.t5 * self.t6 * self.t7
            res_gpu = res_gpu + self.t7 * self.t5 - self.t6 * self.t7
        gpu_time = time.time() - cur

        # The results must be correct and the GPU should be faster
        assert (res_cpu == res_gpu.cpu()).all()
        assert (res_cpu.cpu() == res_gpu.cpu()).all()
        assert (res_cpu.cpu().cpu() == res_gpu.cpu().gpu().cpu()).all()
        assert (res_cpu.gpu() == res_gpu.gpu().gpu()).all()
        assert (res_cpu.gpu() == res_gpu.cpu().gpu()).all()
        assert (res_cpu.cpu().gpu() == res_gpu.gpu()).all()
        assert (res_cpu.cpu() == res_gpu.gpu().cpu()).all()
        print(f"\nops: CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0

    def test_tensor_ops_rng(self):
        """Testing `T` class on random number generation in normal,
           random, and uniformly random distributions. Comparing performance on
           GPU and verifying that it is faster.
        """

        # GPU
        cur = time.time()
        gpu_x = T.rand(self._shape, gpu=True)
        gpu_time = time.time() - cur

        # CPU
        cur = time.time()
        cpu_x = T.rand(self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"\n`rand`: CPU: {cpu_time}, GPU: {gpu_time}")
        # Half a second better already
        assert cpu_time - gpu_time > 0.5

        # GPU
        cur = time.time()
        gpu_x = T.uniform(self._shape, 10, 15, gpu=True)
        gpu_time = time.time() - cur

        # CPU
        cur = time.time()
        cpu_x = T.uniform(self._shape, 10, 15, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"`uniform`: CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0.5

        # GPU
        cur = time.time()
        gpu_x = T.normal(self._shape, gpu=True)
        gpu_time = time.time() - cur

        # CPU
        cur = time.time()
        cpu_x = T.normal(self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"`normal`: CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0.5

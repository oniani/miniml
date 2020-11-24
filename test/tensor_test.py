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
        assert self.t1 + self.t2 != T([3.01, 3, 3])
        assert self.t1 - self.t2 != T([-1, -0.9999, -1])
        assert self.t1 * self.t2 != T([2, 2, 1.9999999])
        assert self.t1 / self.t2 != T([0.55, 0.5, 0.5])

        # `+`, `-`, `*`, `/`, and `==` operations
        assert self.t1 + self.t2 == T([3, 3, 3])
        assert self.t1 - self.t2 == T([-1, -1, -1])
        assert self.t1 * self.t2 == T([2, 2, 2])
        assert self.t1 / self.t2 == T([0.5, 0.5, 0.5])

        # `==`, dot`, and `@`
        assert T.dot(self.t3, self.t4) == T(
            [
                [[22, 28], [58, 64]],
                [[49, 64], [139, 154]],
                [[76, 100], [220, 244]],
            ]
        )
        assert self.t3 @ self.t4 == T(
            [
                [[22, 28], [49, 64], [76, 100]],
                [[58, 64], [139, 154], [220, 244]],
            ]
        )
        self.t3 = self.t3 @ self.t3
        assert self.t3 == T([[30, 36, 42], [66, 81, 96], [102, 126, 150]])

    def test_tensor_ops(self):
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
        assert res_cpu == res_gpu.cpu()
        assert res_cpu.cpu() == res_gpu.cpu()
        assert res_cpu.cpu().cpu() == res_gpu.cpu().gpu().cpu().gpu().cpu()
        assert res_cpu.gpu() == res_gpu.gpu().gpu()
        assert res_cpu == res_gpu
        assert res_cpu.gpu() == res_gpu.cpu()
        assert res_cpu.cpu().cpu() == res_gpu.gpu()
        assert res_cpu.cpu() == res_gpu.cpu().gpu()
        print(f"CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0

    def test_tensor_ops_rng(self):
        """Testing `T` class on random number generation in normal,
           random, and uniformly random distributions. Comparing performance on
           GPU and verifying that it is faster.
        """

        # CPU
        cur = time.time()
        gpu_x = T.rand(self._shape, gpu=True)
        gpu_time = time.time() - cur

        # GPU
        cur = time.time()
        cpu_x = T.rand(self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"CPU: {cpu_time}, GPU: {gpu_time}")
        # Half a second better already
        assert cpu_time - gpu_time > 0.5

        # CPU
        cur = time.time()
        gpu_x = T.uniform(10, 15, self._shape, gpu=True)
        gpu_time = time.time() - cur

        # GPU
        cur = time.time()
        cpu_x = T.uniform(10, 15, self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0.5

        # CPU
        cur = time.time()
        gpu_x = T.normal(self._shape, gpu=True)
        gpu_time = time.time() - cur

        # GPU
        cur = time.time()
        cpu_x = T.normal(self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0.5

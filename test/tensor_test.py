# type: ignore

import time
import pytest

import numpy as np
import pyopencl as cl

from miniml.tensor import Tensor


# Random seed for consistent results
np.random.seed(0)

# Initialize the context
CONTEXT: cl.Context = cl.create_some_context(answers=[0, 1])

# Instantiate a queue
QUEUE: cl.CommandQueue = cl.CommandQueue(CONTEXT)


class TestTensor:
    """Testing the `Tensor` class."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_class(self):
        """Initialize the tensors."""

        self.t1 = Tensor([1, 1, 1])
        self.t2 = Tensor([2, 2, 2])
        self.t3 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.t4 = Tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        )

        self.t5 = Tensor(np.random.rand(150, 150))
        self.t6 = Tensor(np.random.rand(150, 150))
        self.t7 = Tensor(np.random.rand(150, 150))

    def test_tensor_ops(self):
        """Testing `Tensor` class operations on CPU and GPU."""

        # `+`, `-`, `*`, `/`, and `!=` operations
        assert self.t1 + self.t2 != Tensor([3.01, 3, 3])
        assert self.t1 - self.t2 != Tensor([-1, -0.9999, -1])
        assert self.t1 * self.t2 != Tensor([2, 2, 1.9999999])
        assert self.t1 / self.t2 != Tensor([0.55, 0.5, 0.5])

        # `+`, `-`, `*`, `/`, and `==` operations
        assert self.t1 + self.t2 == Tensor([3, 3, 3])
        assert self.t1 - self.t2 == Tensor([-1, -1, -1])
        assert self.t1 * self.t2 == Tensor([2, 2, 2])
        assert self.t1 / self.t2 == Tensor([0.5, 0.5, 0.5])

        # `==`, dot`, and `@`
        assert Tensor.dot(self.t3, self.t4) == Tensor(
            [
                [[22, 28], [58, 64]],
                [[49, 64], [139, 154]],
                [[76, 100], [220, 244]],
            ]
        )
        assert self.t3 @ self.t4 == Tensor(
            [
                [[22, 28], [49, 64], [76, 100]],
                [[58, 64], [139, 154], [220, 244]],
            ]
        )
        self.t3 = self.t3 @ self.t3
        assert self.t3 == Tensor([[30, 36, 42], [66, 81, 96], [102, 126, 150]])

    def test_tensor_gpu(self):
        """Testing `Tensor` class on GPU and verifying that it is faster."""

        # Number of iterations
        its: int = 1000

        # CPU
        cur = time.time()
        res_cpu = Tensor([0])
        for _ in range(its):
            res_cpu = res_cpu + self.t7 * self.t5 * self.t6 * self.t7
        cpu_time = time.time() - cur

        # GPU
        cur = time.time()
        res_gpu = Tensor([0])
        for _ in range(its):
            res_gpu = res_gpu + self.t7 * self.t5 * self.t6 * self.t7
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

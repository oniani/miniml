# type: ignore

import time
import pytest

import numpy as np

import miniml.tensor as T
import miniml.activation as A

# Needed for testing
import pyopencl.clrandom

# Random seed for consistent results
np.random.seed(0)


class TestTensor:
    """Testing the `Tensor` class."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_class(self) -> None:
        """Initialize the tensors."""

        self._t1 = T.Tensor([1, 1, 1])
        self._t2 = T.Tensor([2, 2, 2])
        self._t3 = T.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self._t4 = T.Tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        )

        self._t5 = T.Tensor(np.random.rand(150, 150))
        self._t6 = T.Tensor(np.random.rand(150, 150))
        self._t7 = T.Tensor(np.random.rand(150, 150))
        self._shape = (10_000, 5_000)

        # Tensors
        self._t8 = T.Tensor([-1, 0, 2, 3], gpu=True)
        self._t9 = T.Tensor([-1, 0, 2, 3], gpu=False)

        # Activations
        self._leaky_relu = A.LeakyReLU()
        self._relu = A.ReLU()
        self._tanh = A.Tanh()
        self._sigmoid = A.Sigmoid()
        self._softmax = A.Softmax()
        self._log_softmax = A.LogSoftmax()

    def test_tensor_ops(self) -> None:
        """Testing `T` class operations on CPU and GPU."""

        # `+`, `-`, `*`, `/`, and `!=` operations
        assert (self._t1 + self._t2 != T.Tensor([3.01, 3, 3])).any()
        assert (self._t1 - self._t2 != T.Tensor([-1, -0.9999, -1])).any()
        assert (self._t1 * self._t2 != T.Tensor([2, 2, 1.9999999])).any()
        assert (self._t1 / self._t2 != T.Tensor([0.55, 0.5, 0.5])).any()

        # `+`, `-`, `*`, `/`, and `==` operations
        assert (self._t1 + self._t2 == T.Tensor([3, 3, 3])).all()
        assert (self._t1 - self._t2 == T.Tensor([-1, -1, -1])).all()
        assert (self._t1 * self._t2 == T.Tensor([2, 2, 2])).all()
        assert (self._t1 / self._t2 == T.Tensor([0.5, 0.5, 0.5])).all()

        # `==` and dot`
        assert (
            T.Ops.dot(self._t3, self._t4)
            == T.Tensor(
                [
                    [[22, 28], [58, 64]],
                    [[49, 64], [139, 154]],
                    [[76, 100], [220, 244]],
                ]
            )
        ).all()

    def test_tensor_ops_gpu(self) -> None:
        """Testing `T` class on GPU and verifying that it is faster."""

        # Number of iterations
        its: int = 1_000

        # CPU
        cur = time.time()
        res_cpu = T.Tensor([0])
        for _ in range(its):
            res_cpu = res_cpu + self._t7 * self._t5 * self._t6 * self._t7
            res_cpu = res_cpu + self._t7 * self._t5 - self._t6 * self._t7
        cpu_time = time.time() - cur

        # GPU
        cur = time.time()
        res_gpu = T.Tensor([0])
        for _ in range(its):
            res_gpu = res_gpu + self._t7 * self._t5 * self._t6 * self._t7
            res_gpu = res_gpu + self._t7 * self._t5 - self._t6 * self._t7
        gpu_time = time.time() - cur

        # The results must be correct and the GPU should be faster
        assert (res_cpu == res_gpu.to_cpu()).all()
        assert (res_cpu.to_cpu() == res_gpu.to_cpu()).all()
        assert (
            res_cpu.to_cpu().to_cpu() == res_gpu.to_cpu().to_gpu().to_cpu()
        ).all()
        assert (res_cpu.to_gpu() == res_gpu.to_gpu().to_gpu()).all()
        assert (res_cpu.to_gpu() == res_gpu.to_cpu().to_gpu()).all()
        assert (res_cpu.to_cpu().to_gpu() == res_gpu.to_gpu()).all()
        assert (res_cpu.to_cpu() == res_gpu.to_gpu().to_cpu()).all()
        print(f"\nops: CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0

    def test_tensor_ops_rng(self) -> None:
        """Testing `T` class on random number generation in normal,
           random, and uniformly random distributions. Comparing performance on
           GPU and verifying that it is faster.
        """

        # GPU
        cur = time.time()
        gpu_x = T.Random.rand(self._shape, gpu=True)
        gpu_time = time.time() - cur

        # CPU
        cur = time.time()
        cpu_x = T.Random.rand(self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"\n`rand`: CPU: {cpu_time}, GPU: {gpu_time}")
        # Half a second better already
        assert cpu_time - gpu_time > 0.5

        # GPU
        cur = time.time()
        gpu_x = T.Random.uniform(self._shape, 10, 15, gpu=True)
        gpu_time = time.time() - cur

        # CPU
        cur = time.time()
        cpu_x = T.Random.uniform(self._shape, 10, 15, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"`uniform`: CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0.5

        # GPU
        cur = time.time()
        gpu_x = T.Random.normal(self._shape, gpu=True)
        gpu_time = time.time() - cur

        # CPU
        cur = time.time()
        cpu_x = T.Random.normal(self._shape, gpu=False)
        cpu_time = time.time() - cur

        assert cpu_x.shape == self._shape
        assert gpu_x.shape == self._shape
        assert gpu_x.shape == cpu_x.shape
        print(f"`normal`: CPU: {cpu_time}, GPU: {gpu_time}")
        assert cpu_time - gpu_time > 0.5

    def test_activation(self) -> None:
        """Test activation functions and their derivatives."""

        # Activation functions
        assert (
            self._leaky_relu.forward(self._t8)
            == self._leaky_relu.forward(self._t9).to_gpu()
        )
        assert (
            self._relu.forward(self._t8)
            == self._relu.forward(self._t9).to_gpu()
        )
        assert (
            self._tanh.forward(self._t8)
            == self._tanh.forward(self._t9).to_gpu()
        )
        assert (
            self._sigmoid.forward(self._t8)
            == self._sigmoid.forward(self._t9).to_gpu()
        )
        assert (
            self._softmax.forward(self._t8)
            == self._softmax.forward(self._t9).to_gpu()
        )
        assert (
            self._log_softmax.forward(self._t8)
            == self._log_softmax.forward(self._t9).to_gpu()
        )

        print("\nActivation functions, success")

        # Activation function derivatives
        # assert A.LeakyReLU(self._t8) == A.AD.leaky_relu(self._t9).to_gpu()
        # assert A.ReLU(self._t8) == A.AD.relu(self._t9).to_gpu()
        # assert A.Tanh(self._t8) == A.AD.tanh(self._t9).to_gpu()
        # assert A.Sigmoid(self._t8) == A.AD.sigmoid(self._t9).to_gpu()

        print("Derivatives of `softmax` and `log_softmax` left to implement")

        # assert AD.softmax(t8) == AD.softmax(t9).to_gpu()
        # assert AD.softmax(t8) == AD.softmax(t9).to_gpu()

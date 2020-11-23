# Confer https://arxiv.org/pdf/1412.6980.pdf
# type: ignore

from typing import Callable
from tensor import Tensor

import numpy as np


class Adam:
    """The glorious Adam optimizer."""

    def __init__(
        self,
        stochastic_function: Callable,
        grad_function: Callable,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 10e-8,
    ) -> None:
        """Initialize the variables."""

        self._alpha: float = alpha
        self._beta_1: float = beta_1
        self._beta_2: float = beta_2
        self._epsilon: float = epsilon
        self._stochastic_function: Callable = stochastic_function
        self._grad_function: Callable = grad_function

    def step(self) -> None:
        """Update the state of the optimizer and its parameters."""

        theta_0 = 0
        theta_0_prev = 1
        m_t = 0
        v_t = 0
        t: int = 0

        # While theta not converged do
        while theta_0 != theta_0_prev:
            t += 1

            # Get gradients w.r.t. stochastic objective at timestep t
            g_t = self._grad_function(theta_0)

            # Update biased first moment estimate
            m_t = self._beta_1 * m_t + (1 - self._beta_1) * g_t

            # Update biased second raw moment estimate
            v_t = self._beta_2 * v_t + (1 - self._beta_2) * (g_t * g_t)

            # Compute bias-corrected first moment estimate
            m_cap = m_t / (1 - (self._beta_1 ** t))

            # Compute bias-corrected second raw moment estimate
            v_cap = v_t / (1 - (self._beta_2 ** t))

            # Save the previous state
            theta_0_prev = theta_0

            # Update parameters
            theta_0 = theta_0 - (self._alpha * m_cap) / (
                np.sqrt(v_cap) + self._epsilon
            )

        return theta_0


def fun(x):
    return x * x - 4 * x + 4


def grad_fun(x):
    return 2 * x - 4


if __name__ == "__main__":
    optimizer = Adam(fun, grad_fun)
    print(vars(optimizer))
    print(optimizer.step())

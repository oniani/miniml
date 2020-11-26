## type: ignore
"""
Author: David Oniani

Description: An implementation of the glorious Adam optimizer based on the
             paper "Adam: A Method for Stochastic Optimization." Avaibale at
             https://arxiv.org/abs/1412.6980.
"""


import numpy as np

from typing import Callable

import miniml.tensor as T


class Adam:
    """The glorious Adam optimizer."""

    def __init__(
        self,
        stochastic_function: Callable,
        grad_function: Callable,
        params: list[dict[str, str]],
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the variables."""

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if beta_1 < 0 or beta_1 > 1.0:
            raise ValueError(f"Invalid beta_1 value: {beta_1}")
        if beta_2 < 0 or beta_2 > 1.0:
            raise ValueError(f"Invalid beta_2 value: {beta_2}")

        self._lr: float = lr
        self._beta_1: float = beta_1
        self._beta_2: float = beta_2
        self._eps: float = eps
        self._stochastic_function: Callable = stochastic_function
        self._grad_function: Callable = grad_function
        self._paramas: list[dict[str, str]] = params

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
            theta_0 = theta_0 - (self._lr * m_cap) / (
                np.sqrt(v_cap) + self._eps  # type: ignore
            )

        return theta_0  # type: ignore


def fun(x: T.Tensor):
    return x * x - 4 * x + 4


def grad_fun(x: T.Tensor):
    return 2 * x - 4


if __name__ == "__main__":
    from nn import Net  # type: ignore

    model = Net()
    optimizer = Adam(fun, grad_fun, model._params)
    print(vars(optimizer))
    print(optimizer.step())

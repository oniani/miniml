⚠

The project is under development. Some core features (e.g., GPU array
broadcasting) are yet to be implemented.

# miniml

A minimal ML library with OpenCL GPU support.

## Perf

| Category  | CPU   | GPU   | Perf Boost |
| --------- | ----- | ----- | ---------- |
| `rand`    | 0.747 | 0.002 | 37X        |
| `uniform` | 0.820 | 0.002 | 41X        |
| `normal`  | 1.829 | 0.002 | 91X        |

## Features

### Activation Functions

- [x] ReLU
- [x] LeakyReLU
- [x] Tanh
- [x] Sigmoid
- [ ] Softmax
- [ ] LogSoftmax

### Tensor

Most important operations are implemented. They work on both CPU and GPU.
The API is very simple:

```python
import miniml.tensor as T

# Initialize a CPU tensor
t1 = T.Tensor([1, 2, 3])

# Initialize a GPU tensor
t2 = T.Tensor([1, 2, 3], gpu=True)

# Load the GPU tensor onto the CPU
t2 = t2.to_gpu()

# Load it back onto the GPU
t2 = t2.to_gpu()
```

CPU tensors use NumPy operations and the GPU tensors use PyOpenCL
operations.

## Tests

```console
python3 -m pytest -sv
```

---

TODO:

- Implement a reverse-mode automatic differentiation
- Implement network layers
- Implement loss function(s?)
- Implement Adam optimizer
- More tests (maybe use a monadic test-generator like Hypothesis?)

## References

- [NumPy](https://numpy.org/)
- [OpenCL](https://www.khronos.org/opencl/)
- [PyOpenCL](https://documen.tician.de/pyopencl/index.html)

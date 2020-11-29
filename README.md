# miniml

A minimal machine learning library, implemented from scratch with OpenCL GPU
support.

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

```
import miniml.tensor as T

# Initialize a CPU tensor
t1 = T.Tensor([1, 2, 3])

# Initialize a GPU tensor
t2 = T.Tensor([1, 2, 3], gpu=True)

# Convert a CPU tensor to a GPU tensor
t1 = t1.to_gpu()

# Convert it back to CPU
t1 = t1.to_cpu()
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

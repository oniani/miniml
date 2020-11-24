# miniml

A minimal machine learning library using the reverse mode automatic
differentiation - implemented from scratch, with OpenCL GPU support.

## perf

- ops: CPU: 0.08780694007873535, GPU: 0.08255696296691895
- `rand`: CPU: 0.7474880218505859, GPU: 0.0021669864654541016
- `uniform`: CPU: 0.8201138973236084, GPU: 0.0024979114532470703
- `normal`: CPU: 1.8289899826049805, GPU: 0.002346038818359375

---

TODO:

- Implement a reverse-mode automatic differentiation
- More tests
- Implement network layers
- Implement a loss function(s?)
- Implement Adam optimizer

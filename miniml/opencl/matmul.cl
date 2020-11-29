// This is a rather naive implementation, not the state-of-the-art GEMM, but is
// fast nevertheless and good enough for the library
//
// A = M x N
// B = N x O
// C = M x O
//
__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global       float* C,
                     const int M,
                     const int O,
                     const int N)
{
    // From 0 to M - 1
    // Global position in x direction
    const int row = get_global_id(0);

    // From 0 to O - 1
    // Global position in y direction
    const int col = get_global_id(1);

    // Accumulator
    float acc = 0.0f;

    // Compute a single element of the resulting matrix
    for (int k = 0; k < N; k++) {
        acc += A[row * N + k] * B[k * O + col];
    }

    // Fill the matrix
    C[row * O + col] = acc;
}

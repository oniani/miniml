import pyopencl as cl


def binary_op_prog(ctx, fn, op):
    """A program that performs a binary operation."""

    code = f"""
         __kernel void {fn}(__global const float *a_g,
                            __global const float *b_g,
                            __global       float *res_g)
         {{
           int gid = get_global_id(0);
           res_g[gid] = a_g[gid] {op} b_g[gid];
         }}
    """

    return cl.Program(ctx, code).build()

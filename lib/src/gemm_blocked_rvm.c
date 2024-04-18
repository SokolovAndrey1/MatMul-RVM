#include "gemm.h"

#define BLOCK_SIZE 4

/**
 * Multiplies two matrices A and B with dimensions n x m and m x k respectively
 * using a block-based approach and THEAD RISC-V matrix extension.
 *
 * @param A Pointer to the first matrix (size n x m).
 * @param B Pointer to the second matrix (size m x k).
 * @param C Pointer to the resulting matrix (size n x k).
 * @param n Number of rows in matrix A and resulting matrix C.
 * @param m Number of columns in matrix A and number of rows in matrix B.
 * @param k Number of columns in matrix B and resulting matrix C.
 */
extern void gemm_block4x4_rvm(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k)
{
    // TODO: Implement Gemm using THEAD RISC-V matrix extension.
    gemm_ref(A, B, C, n, m, k);    
}

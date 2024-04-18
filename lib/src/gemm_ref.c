#include "gemm.h"

/**
 * Multiplies two matrices.
 *
 * @param A Pointer to the first matrix (size n x m).
 * @param B Pointer to the second matrix (size m x k).
 * @param C Pointer to the resulting matrix (size n x k).
 * @param n Number of rows in matrix A and resulting matrix C.
 * @param m Number of columns in matrix A and number of rows in matrix B.
 * @param k Number of columns in matrix B and resulting matrix C.
 */
extern void gemm_ref(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k)
{
    for (size_t i = 0; i < n; i++) { /* Loop over the rows of C */
        for (size_t j = 0; j < k; j++) { /* Loop over the columns of C */
            for (size_t p = 0; p < m; p++) {  /* Update C( i,j ) with the inner
                                                 product of the ith row of A and
                                                 the jth column of B */
                C[i * k + j] += A[i * m + p] * B[p * k + j];
            }
        }
    }
}

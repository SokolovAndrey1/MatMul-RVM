#include "gemm.h"

#define BLOCK_SIZE 4

static inline void process_block_4x4(const size_t k, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc);

/**
 * Multiplies two matrices A and B with dimensions n x m and m x k respectively
 * using a block-based approach.
 *
 * @param A Pointer to the first matrix (size n x m).
 * @param B Pointer to the second matrix (size m x k).
 * @param C Pointer to the resulting matrix (size n x k).
 * @param n Number of rows in matrix A and resulting matrix C.
 * @param m Number of columns in matrix A and number of rows in matrix B.
 * @param k Number of columns in matrix B and resulting matrix C.
 */
extern void gemm_block4x4_ref(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k)
{
    for (size_t j = 0; j < k; j += BLOCK_SIZE) { /* Loop over the columns of C */
        for (size_t i = 0; i < n; i += BLOCK_SIZE) { /* Loop over the rows of C */
            process_block_4x4(m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
}

static inline void process_block_4x4(const size_t k, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc) {
    for (size_t i = 0; i < BLOCK_SIZE; i += 1) {
        for (size_t j = 0; j < BLOCK_SIZE; j += 1) {
            for (size_t p = 0; p < k; p += 1) {
                C[(j * ldc) + i] += A[(j * lda) + p] * B[(p * ldb) + i];
            }
        }
    }
}

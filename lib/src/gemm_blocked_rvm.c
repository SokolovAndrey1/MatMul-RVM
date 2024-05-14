#include "gemm.h"

#define BLOCK_SIZE 4

static inline void process_block_4x4(const size_t k, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc, float *BT);

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
extern void gemm_block4x4_rvm(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k) {
    mcfgm(BLOCK_SIZE);
    mcfgn(BLOCK_SIZE);
    mcfgk(BLOCK_SIZE * sizeof(*A));
    float *BT = malloc(BLOCK_SIZE * m * sizeof(*B));
    for (size_t j = 0; j < k; j += BLOCK_SIZE) {     /* Loop over the columns of C */
        for (size_t i = 0; i < n; i += BLOCK_SIZE) { /* Loop over the rows of C */
            process_block_4x4(m, &A[i * m], m, &B[j], k, &C[(i * k) + j], k, BT);
        }
    }
}

static inline void process_block_4x4(const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc, float *BT) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < BLOCK_SIZE; j++) {
            BT[j * m + i] = B[i * ldb + j];
        }
    }
    mfloat32_t ma;
    mfloat32_t mb;
    mfloat32_t ans = mld_f32(C, ldc * sizeof(*C));
    for (size_t p = 0; p < m; p += BLOCK_SIZE) {
        ma = mld_f32(A + p, lda * sizeof(*A));
        mb = mld_f32(BT + p, lda * sizeof(*B));
        ans = fmmacc_mf32(ans, ma, mb);
        mst_f32_mf32(C, ldc * sizeof(*C), ans);
    }
}

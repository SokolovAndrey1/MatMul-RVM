#include "gemm.h"
#include <riscv_matrix.h>


#define BLOCK_SIZE 4

static inline void process_block_4x4(const size_t k, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc);

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
    //gemm_ref(A, B, C, n, m, k);    
    for (size_t j = 0; j < k; j += BLOCK_SIZE) { /* Loop over the columns of C */
        for (size_t i = 0; i < n; i += BLOCK_SIZE) { /* Loop over the rows of C */
            process_block_4x4(m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
}

static inline void process_block_4x4(const size_t k, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc) {
    uint16_t m_k = BLOCK_SIZE * sizeof(float32_t);

    uint8_t row_size = 16;
    long stride = row_size * sizeof(float32_t);

    mcfgm(BLOCK_SIZE); 
    mcfgn(BLOCK_SIZE); 
    mcfgk(m_k); 

    mfloat32_t ma = mld_f32(A, stride);
    mfloat32_t mb = mld_f32(B, stride);
    mfloat32_t ans = mzero_mf32();

    for (int i = 0; i < 4; ++i) {
        mfloat32_t mb_vector = mmov_mf32v(mb, i);
        ans = fmmacc_mf32(ans, ma, mb_vector);
    }
    mst_f32_mf32(C, 16, ans);
}

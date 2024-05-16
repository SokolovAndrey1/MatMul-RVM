#include "gemm.h"

#define BLOCK_SIZE 4

static inline void process_block_nxk(const size_t n, const size_t k, const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc);

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
    size_t blocks_n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    size_t blocks_k = (k / BLOCK_SIZE) * BLOCK_SIZE;

    size_t tail_n = n - blocks_n;
    size_t tail_k = k - blocks_k;

    size_t i;
    size_t j;
    for (i = 0; i < blocks_n; i += BLOCK_SIZE) {
        for (j = 0; j < blocks_k; j += BLOCK_SIZE) {
            process_block_nxk(BLOCK_SIZE, BLOCK_SIZE, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
        if (tail_k != 0) {
            process_block_nxk(BLOCK_SIZE, tail_k, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
    if (tail_n != 0) {
        for (j = 0; j < blocks_k; j += BLOCK_SIZE) {
            process_block_nxk(tail_n, BLOCK_SIZE, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
        if (tail_k != 0){
            process_block_nxk(tail_n, tail_k, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
}

static inline void process_block_nxk(const size_t n, const size_t k, const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc) {
    float block_b[BLOCK_SIZE * BLOCK_SIZE];

    const size_t blocks_m = (m / BLOCK_SIZE) * BLOCK_SIZE;
    const size_t tail_m = m - blocks_m;

    const float *ptr_a = A;
    const float *ptr_b = B;

    const size_t stride_a = lda * sizeof(float);

    // To increment B pointer after each iteration
    const size_t offset_b = BLOCK_SIZE * ldb;

    mcfgm(n);
    mcfgk(k * sizeof(float));
    // Load C (n x k) block
    mfloat32_t mc = mld_f32(C, ldc * sizeof(float));

    size_t block_m;
    for (block_m = 0; block_m < blocks_m; block_m += BLOCK_SIZE) {
        mcfgm(n);
        mcfgk(BLOCK_SIZE * sizeof(float));
        // Load A (n x 4) block
        mfloat32_t ma = mld_f32(A+block_m, stride_a);

        // Transpose B (4 x k) block
        for (int row = 0; row < BLOCK_SIZE; row++) {
            for (int col = 0; col < k; col++) {
                block_b[col*BLOCK_SIZE+row] = ptr_b[(row * ldb) + col];
            }
        }

        mcfgm(k);
        // Load B (k x 4) block
        mfloat32_t mb = mld_f32(block_b, BLOCK_SIZE * sizeof(float));

        mcfgm(n);
        mcfgn(k);
        // C (n x k) += A (n x 4) * B (k x 4)
        mc = fmmacc_mf32(mc, ma, mb);

        ptr_b += offset_b;
        ptr_a += BLOCK_SIZE;
    }
    if (tail_m) {
        mcfgm(n);
        mcfgn(BLOCK_SIZE);
        mcfgk(tail_m * sizeof(float));
        // Load A (n x tail_m) block
        mfloat32_t ma = mld_f32(A+block_m, stride_a);

        // Transpose B (tail_m x k) block
        for (int row = 0; row < tail_m; row++) {
            for (int col = 0; col < k; col++) {
                block_b[col*BLOCK_SIZE+row] = ptr_b[(row * ldb) + col];
            }
        }

        mcfgm(k);
        // Load B (k x tail_m) block
        mfloat32_t mb = mld_f32(block_b, BLOCK_SIZE * sizeof(float));

        mcfgm(n);
        mcfgn(k);
        // C (n x k) += A (tail_m x 4) * B (tail_m x 4)
        mc = fmmacc_mf32(mc, ma, mb);
    }
    mcfgk(k * sizeof(float));
    // Store C (n x k) block
    mst_f32_mf32(C, ldc * sizeof(float), mc);
}

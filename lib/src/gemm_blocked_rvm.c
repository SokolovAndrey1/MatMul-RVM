#include "gemm.h"
#include <riscv_matrix.h>

#define BLOCK_SIZE 4

static inline void process_block_4x4(const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc) {
    long stride = k * sizeof(float32_t);
    printf("k = %d stride = %d\n", k, stride);

    // загрузка блоков    
    mfloat32_t ma = mld_f32(A, stride);
    mfloat32_t mb = mld_f32(B, stride);
    mfloat32_t mc = mld_f32(C, stride);

    //print_matrix_reg("|ma|", ma, k, k);
    //print_matrix_reg("|mb|", mb, k, k);
    //print_matrix_reg("|mc|", mc, k, k);

    
    // вычисление результата
    mfloat32_t res = fmmacc_mf32(mc, ma, mb);

    print_matrix_reg("|iteration res|", res, k, k);

    // выгрузка
    mst_f32_mf32(C, stride, res);
}

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
    //print_matrix(A, n, m);
    //print_matrix(B, m, k);

    
    // конфигурация матричных регистров
    mcfgm(BLOCK_SIZE);
    mcfgn(BLOCK_SIZE);
    mcfgk(BLOCK_SIZE * sizeof(float32_t));
    
    for (size_t j = 0; j < k; j += BLOCK_SIZE) { /* Loop over the columns of C */
        for (size_t i = 0; i < n; i += BLOCK_SIZE) { /* Loop over the rows of C */
            //printf("i = %d, j = %d, m = %d, k = %d\n", i,j,m,k);
            process_block_4x4(m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
    
    //printf("rvm:\n");
    //print_matrix(C, n, k);
}

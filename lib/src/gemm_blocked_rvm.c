#include "gemm.h"
#include <riscv_matrix.h>

#define BLOCK_SIZE 4

static inline void process_block_4x4(const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc) {
    // загрузка блоков    
    mfloat32_t ma = mld_f32(A, lda * sizeof(*A));
    mfloat32_t mb = mld_f32(B, ldb * sizeof(*B));
    mfloat32_t mc = mld_f32(C, ldc * sizeof(float));

    //print_matrix_reg("|ma|", ma, m, m);
    //print_matrix_reg("|mb|", mb, m, m);
    //print_matrix_reg("|mc|", mc, m, m);
    
    // вычисление результата
    mfloat32_t res = fmmacc_mf32(mc, ma, mb);

    //print_matrix_reg("|iteration res|", res, m, m);

    // выгрузка
    mst_f32_mf32(C, ldc * sizeof(*C), res);
}

static inline void transpose_4x4(const float *B, float *T, const size_t m, const size_t k) {
    for(size_t i = 0; i < m; ++i)
        for(size_t j = 0; j < k; ++j)
            T[j*m+i] = B[i*k+j];
    
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
    // транспонирование матрицы B
    float T[sizeof(B)] = {0};
    transpose_4x4(B, T, m, k);
    
    //print_matrix(B, m, k);
    //print_matrix(T, k, m);
    
    // конфигурация матричных регистров
    mcfgm(BLOCK_SIZE);
    mcfgn(BLOCK_SIZE);
    mcfgk(BLOCK_SIZE * sizeof(float));
    
    for (size_t j = 0; j < k; j += BLOCK_SIZE) { /* Loop over the columns of C */
        for (size_t i = 0; i < n; i += BLOCK_SIZE) { /* Loop over the rows of C */
            process_block_4x4(m, &A[i*m], m, &T[j], k, &C[(i*k) + j], k);
        }
    }
    
    ////printf("rvm:\n");
    ////print_matrix(C, n, k);
}

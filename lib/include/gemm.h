#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include <riscv_matrix.h>
#include <riscv_vector.h>

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
extern void gemm_ref(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k);


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
extern void gemm_block4x4_ref(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k);

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
extern void gemm_block4x4_rvv(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k);

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
extern void gemm_block4x4_rvm(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k);

//
// Utils functions
//
void print_matrix(float* A, size_t n, size_t m);
void print_matrix_reg(const char *fmt, const mfloat32_t reg, const size_t n, const size_t m);

#endif // GEMM_H

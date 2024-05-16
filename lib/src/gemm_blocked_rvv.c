#include "gemm.h"

#define BLOCK_SIZE 4

static inline void process_block_nx4(const size_t n, const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc);
static inline void process_block_nxk(const size_t n, const size_t k, const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc);

/**
 * Multiplies two matrices A and B with dimensions n x m and m x k respectively
 * using a block-based approach and RISC-V Vector extension.
 *
 * @param A Pointer to the first matrix (size n x m).
 * @param B Pointer to the second matrix (size m x k).
 * @param C Pointer to the resulting matrix (size n x k).
 * @param n Number of rows in matrix A and resulting matrix C.
 * @param m Number of columns in matrix A and number of rows in matrix B.
 * @param k Number of columns in matrix B and resulting matrix C.
 */
extern void gemm_block4x4_rvv(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k)
{
    size_t blocks_n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    size_t blocks_k = (k / BLOCK_SIZE) * BLOCK_SIZE;

    size_t tail_n = n - blocks_n;
    size_t tail_k = k - blocks_k;

    size_t j;
    size_t i;
    for (i = 0; i < blocks_n; i += BLOCK_SIZE)
    {
        for (j = 0; j < blocks_k; j += BLOCK_SIZE)
        {
            process_block_nx4(BLOCK_SIZE, m, &A[i * m], m, &B[j], k, &C[(i * k) + j], k);
        }
        if (tail_k != 0)
        {
            process_block_nxk(BLOCK_SIZE, tail_k, m, &A[i * m], m, &B[j], k, &C[(i * k) + j], k);
        }
    }
    if (tail_n != 0)
    {
        for (j = 0; j < blocks_k; j += BLOCK_SIZE)
        {
            process_block_nx4(tail_n, m, &A[i * m], m, &B[j], k, &C[(i * k) + j], k);
        }
        if (tail_k != 0)
        {
            process_block_nxk(tail_n, tail_k, m, &A[i * m], m, &B[j], k, &C[(i * k) + j], k);
        }
    }
}

static inline void process_block_nxk(const size_t n, const size_t k, const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc)
{
    assert(k < 4);

    const float *ptr_a = A;
    const float *ptr_b = B;

    size_t blocks = (m / BLOCK_SIZE) * BLOCK_SIZE;
    size_t tail = m - blocks;

    for (size_t rem = blocks; rem > 0; rem -= BLOCK_SIZE)
    {
        vfloat32m1_t vec_a00_a10_a20_a30 = vlse32_v_f32m1(ptr_a + 0, lda * sizeof(float), n);
        vfloat32m1_t vec_a01_a11_a21_a31 = vlse32_v_f32m1(ptr_a + 1, lda * sizeof(float), n);
        vfloat32m1_t vec_a02_a12_a32_a32 = vlse32_v_f32m1(ptr_a + 2, lda * sizeof(float), n);
        vfloat32m1_t vec_a03_a13_a23_a33 = vlse32_v_f32m1(ptr_a + 3, lda * sizeof(float), n);

        for (size_t k_offset = 0; k_offset < k; k_offset += 1)
        {
            const float *ptr_b_col = ptr_b + k_offset;
            const float *ptr_c = C + k_offset;

            vfloat32m1_t vec_C_col = vlse32_v_f32m1(ptr_c, ldc * sizeof(float), n);

            vfloat32m1_t b00 = vfmv_v_f_f32m1(*(ptr_b_col + 0 * ldb), n);
            vfloat32m1_t b10 = vfmv_v_f_f32m1(*(ptr_b_col + 1 * ldb), n);
            vfloat32m1_t b20 = vfmv_v_f_f32m1(*(ptr_b_col + 2 * ldb), n);
            vfloat32m1_t b30 = vfmv_v_f_f32m1(*(ptr_b_col + 3 * ldb), n);

            vec_C_col = vfmacc_vv_f32m1(vec_C_col, b00, vec_a00_a10_a20_a30, n);
            vec_C_col = vfmacc_vv_f32m1(vec_C_col, b10, vec_a01_a11_a21_a31, n);
            vec_C_col = vfmacc_vv_f32m1(vec_C_col, b20, vec_a02_a12_a32_a32, n);
            vec_C_col = vfmacc_vv_f32m1(vec_C_col, b30, vec_a03_a13_a23_a33, n);

            vsse32_v_f32m1(ptr_c, ldc * sizeof(float), vec_C_col, n);
        }

        const size_t offset = BLOCK_SIZE * ldb;
        ptr_b += offset;
        ptr_a += BLOCK_SIZE;
    }

    for (size_t rem = tail; rem > 0; rem -= 1)
    {
        vfloat32m1_t vec_a00_a10_a20_a30 = vlse32_v_f32m1(ptr_a + 0, lda * sizeof(float), n);

        for (size_t k_offset = 0; k_offset < k; k_offset += 1)
        {
            const float *ptr_b_col = ptr_b + k_offset;
            const float *ptr_c = C + k_offset;

            vfloat32m1_t vec_C_col = vlse32_v_f32m1(ptr_c, ldc * sizeof(float), n);

            vfloat32m1_t b00 = vfmv_v_f_f32m1(*(ptr_b_col + 0 * ldb), n);

            vec_C_col = vfmacc_vv_f32m1(vec_C_col, b00, vec_a00_a10_a20_a30, n);

            vsse32_v_f32m1(ptr_c, ldc * sizeof(float), vec_C_col, n);
        }
        ptr_a += 1;
        ptr_b += ldb;
    }
}

static inline void process_block_nx4(const size_t n, const size_t m, const float *A, const size_t lda, const float *B, const size_t ldb, float *C, const size_t ldc)
{

    const float *ptr_a = A;
    const float *ptr_b = B;
    size_t blocks = (m / BLOCK_SIZE) * BLOCK_SIZE;
    size_t tail = m - blocks;

    vfloat32m1_t sum_c00_c10_c20_c30 = vfmv_v_f_f32m1(0, n);
    vfloat32m1_t sum_c01_c11_c21_c31 = vfmv_v_f_f32m1(0, n);
    vfloat32m1_t sum_c02_c12_c22_c32 = vfmv_v_f_f32m1(0, n);
    vfloat32m1_t sum_c03_c13_c23_c33 = vfmv_v_f_f32m1(0, n);

    for (size_t rem = blocks; rem > 0; rem -= BLOCK_SIZE)
    {
        vfloat32m1_t vec_a00_a10_a20_a30 = vlse32_v_f32m1(ptr_a + 0, lda * sizeof(float), n);
        vfloat32m1_t vec_a01_a11_a21_a31 = vlse32_v_f32m1(ptr_a + 1, lda * sizeof(float), n);
        vfloat32m1_t vec_a02_a12_a32_a32 = vlse32_v_f32m1(ptr_a + 2, lda * sizeof(float), n);
        vfloat32m1_t vec_a03_a13_a23_a33 = vlse32_v_f32m1(ptr_a + 3, lda * sizeof(float), n);

        vfloat32m1_t b00 = vfmv_v_f_f32m1(*(ptr_b + 0), n);
        vfloat32m1_t b01 = vfmv_v_f_f32m1(*(ptr_b + 1), n);
        vfloat32m1_t b02 = vfmv_v_f_f32m1(*(ptr_b + 2), n);
        vfloat32m1_t b03 = vfmv_v_f_f32m1(*(ptr_b + 3), n);

        vfloat32m1_t b10 = vfmv_v_f_f32m1(*(ptr_b + 1 * ldb + 0), n);
        vfloat32m1_t b11 = vfmv_v_f_f32m1(*(ptr_b + 1 * ldb + 1), n);
        vfloat32m1_t b12 = vfmv_v_f_f32m1(*(ptr_b + 1 * ldb + 2), n);
        vfloat32m1_t b13 = vfmv_v_f_f32m1(*(ptr_b + 1 * ldb + 3), n);

        vfloat32m1_t b20 = vfmv_v_f_f32m1(*(ptr_b + 2 * ldb + 0), n);
        vfloat32m1_t b21 = vfmv_v_f_f32m1(*(ptr_b + 2 * ldb + 1), n);
        vfloat32m1_t b22 = vfmv_v_f_f32m1(*(ptr_b + 2 * ldb + 2), n);
        vfloat32m1_t b23 = vfmv_v_f_f32m1(*(ptr_b + 2 * ldb + 3), n);

        vfloat32m1_t b30 = vfmv_v_f_f32m1(*(ptr_b + 3 * ldb + 0), n);
        vfloat32m1_t b31 = vfmv_v_f_f32m1(*(ptr_b + 3 * ldb + 1), n);
        vfloat32m1_t b32 = vfmv_v_f_f32m1(*(ptr_b + 3 * ldb + 2), n);
        vfloat32m1_t b33 = vfmv_v_f_f32m1(*(ptr_b + 3 * ldb + 3), n);

        sum_c00_c10_c20_c30 = vfmacc_vv_f32m1(sum_c00_c10_c20_c30, b00, vec_a00_a10_a20_a30, n);
        sum_c01_c11_c21_c31 = vfmacc_vv_f32m1(sum_c01_c11_c21_c31, b01, vec_a00_a10_a20_a30, n);
        sum_c02_c12_c22_c32 = vfmacc_vv_f32m1(sum_c02_c12_c22_c32, b02, vec_a00_a10_a20_a30, n);
        sum_c03_c13_c23_c33 = vfmacc_vv_f32m1(sum_c03_c13_c23_c33, b03, vec_a00_a10_a20_a30, n);

        sum_c00_c10_c20_c30 = vfmacc_vv_f32m1(sum_c00_c10_c20_c30, b10, vec_a01_a11_a21_a31, n);
        sum_c01_c11_c21_c31 = vfmacc_vv_f32m1(sum_c01_c11_c21_c31, b11, vec_a01_a11_a21_a31, n);
        sum_c02_c12_c22_c32 = vfmacc_vv_f32m1(sum_c02_c12_c22_c32, b12, vec_a01_a11_a21_a31, n);
        sum_c03_c13_c23_c33 = vfmacc_vv_f32m1(sum_c03_c13_c23_c33, b13, vec_a01_a11_a21_a31, n);

        sum_c00_c10_c20_c30 = vfmacc_vv_f32m1(sum_c00_c10_c20_c30, b20, vec_a02_a12_a32_a32, n);
        sum_c01_c11_c21_c31 = vfmacc_vv_f32m1(sum_c01_c11_c21_c31, b21, vec_a02_a12_a32_a32, n);
        sum_c02_c12_c22_c32 = vfmacc_vv_f32m1(sum_c02_c12_c22_c32, b22, vec_a02_a12_a32_a32, n);
        sum_c03_c13_c23_c33 = vfmacc_vv_f32m1(sum_c03_c13_c23_c33, b23, vec_a02_a12_a32_a32, n);

        sum_c00_c10_c20_c30 = vfmacc_vv_f32m1(sum_c00_c10_c20_c30, b30, vec_a03_a13_a23_a33, n);
        sum_c01_c11_c21_c31 = vfmacc_vv_f32m1(sum_c01_c11_c21_c31, b31, vec_a03_a13_a23_a33, n);
        sum_c02_c12_c22_c32 = vfmacc_vv_f32m1(sum_c02_c12_c22_c32, b32, vec_a03_a13_a23_a33, n);
        sum_c03_c13_c23_c33 = vfmacc_vv_f32m1(sum_c03_c13_c23_c33, b33, vec_a03_a13_a23_a33, n);

        ptr_a += BLOCK_SIZE;
        const size_t offset = BLOCK_SIZE * ldb;
        ptr_b += offset;
    }
    for (size_t rem = tail; rem > 0; rem -= 1)
    {
        vfloat32m1_t vec_a00_a01_a02_a03 = vlse32_v_f32m1(ptr_a + 0, lda * sizeof(float), n);

        vfloat32m1_t b00 = vfmv_v_f_f32m1(*(ptr_b + 0), n);
        vfloat32m1_t b01 = vfmv_v_f_f32m1(*(ptr_b + 1), n);
        vfloat32m1_t b02 = vfmv_v_f_f32m1(*(ptr_b + 2), n);
        vfloat32m1_t b03 = vfmv_v_f_f32m1(*(ptr_b + 3), n);

        sum_c00_c10_c20_c30 = vfmacc_vv_f32m1(sum_c00_c10_c20_c30, b00, vec_a00_a01_a02_a03, n);
        sum_c01_c11_c21_c31 = vfmacc_vv_f32m1(sum_c01_c11_c21_c31, b01, vec_a00_a01_a02_a03, n);
        sum_c02_c12_c22_c32 = vfmacc_vv_f32m1(sum_c02_c12_c22_c32, b02, vec_a00_a01_a02_a03, n);
        sum_c03_c13_c23_c33 = vfmacc_vv_f32m1(sum_c03_c13_c23_c33, b03, vec_a00_a01_a02_a03, n);

        ptr_a += 1;
        const size_t offset = 1 * ldb;
        ptr_b += offset;
    }

    vfloat32m1_t vec_c00_c10_c20_c30 = vlse32_v_f32m1(C + 0, ldc * sizeof(float), n);
    vfloat32m1_t vec_c01_c11_c21_c31 = vlse32_v_f32m1(C + 1, ldc * sizeof(float), n);
    vfloat32m1_t vec_c02_c12_c22_c32 = vlse32_v_f32m1(C + 2, ldc * sizeof(float), n);
    vfloat32m1_t vec_c03_c13_c23_c33 = vlse32_v_f32m1(C + 3, ldc * sizeof(float), n);

    vec_c00_c10_c20_c30 = vfadd_vv_f32m1(vec_c00_c10_c20_c30, sum_c00_c10_c20_c30, n);
    vec_c01_c11_c21_c31 = vfadd_vv_f32m1(vec_c01_c11_c21_c31, sum_c01_c11_c21_c31, n);
    vec_c02_c12_c22_c32 = vfadd_vv_f32m1(vec_c02_c12_c22_c32, sum_c02_c12_c22_c32, n);
    vec_c03_c13_c23_c33 = vfadd_vv_f32m1(vec_c03_c13_c23_c33, sum_c03_c13_c23_c33, n);

    vsse32_v_f32m1(C + 0, ldc * sizeof(float), vec_c00_c10_c20_c30, n);
    vsse32_v_f32m1(C + 1, ldc * sizeof(float), vec_c01_c11_c21_c31, n);
    vsse32_v_f32m1(C + 2, ldc * sizeof(float), vec_c02_c12_c22_c32, n);
    vsse32_v_f32m1(C + 3, ldc * sizeof(float), vec_c03_c13_c23_c33, n);
}

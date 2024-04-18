#include "gemm.h"

/**
 * Prints the contents of a 2D matrix.
 * 
 * @param A Pointer to the 1D array representing the matrix.
 * @param n Number of rows in the matrix.
 * @param m Number of columns in the matrix.
 */
void print_matrix(float* A, size_t n, size_t m) {
    for (size_t row = 0; row < n; row++)
    {
        for (size_t col = 0; col < m; col++)
        {
            printf("%.3f ", A[row * m + col]);
        }
        printf("\n");
    }
    printf("\n");
}

#ifdef RV64GVM
/**
 * Prints the contents of a matrix stored in a matrix register using the given format.
 * 
 * @param fmt Header of output.
 * @param reg float matrix register containing the matrix elements.
 * @param n Number of rows in the matrix.
 * @param m Number of columns in the matrix.
 */
void print_matrix_reg(const char *fmt, mfloat32_t reg, size_t n, size_t m)
{
    // const size_t max_reg_count = 16;
    float tmp_buf[16] = {0.0f};
    printf("%s:\n", fmt);
    mst_f32_mf32(tmp_buf, n * sizeof(float), reg);
    for (size_t row = 0; row < n; row++)
    {
        for (size_t col = 0; col < m; col++)
        {
            printf("%.2f ", tmp_buf[row * n + col]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif // RV64GVM
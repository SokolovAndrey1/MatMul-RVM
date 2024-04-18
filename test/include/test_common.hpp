#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

extern "C" {
#include "gemm.h"
}

#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

/**
 * @brief Compares two matrices element-wise for equality within a specified relative error.
 *
 * This function compares each element of the expected and actual matrices within a specified relative error.
 *
 * @tparam T The data type of the matrix elements.
 * @param expected Pointer to the expected matrix.
 * @param actual Pointer to the actual matrix.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 * @param relative_error The maximum allowable relative error between corresponding elements.
 *                       Defaults to 1e-6 if not specified.
 * @return Assertion success if the matrices are equal within the specified relative error,
 *         or assertion failure with a detailed error message otherwise.
 */
template <typename T>
::testing::AssertionResult AssertMatricesEqual(const T* expected, const T* actual, size_t rows, size_t cols, double relative_error = 1e-6) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            T abs_expected = std::abs(expected[i * cols + j]);
            T abs_actual = std::abs(actual[i * cols + j]);

            T abs_diff = std::abs(expected[i * cols + j] - actual[i * cols + j]);

            T min = std::min(abs_expected, abs_actual);
            if (min == T{0.0}) {
                const T zeroTolerance = 1e-13;
                if (abs_diff > zeroTolerance) {
                    std::stringstream message;
                    message << "ABS difference at position (" << i << ", " << j << ") is " << std::setprecision(6) << abs_diff
                            << ", which exceeds the abs error " << std::setprecision(6) << zeroTolerance
                            << ". Expected: " << std::setprecision(6) << expected[i * cols + j]
                            << ", Actual: " << std::setprecision(6) << actual[i * cols + j];
                    return ::testing::AssertionFailure() << message.str();
                }
                else
                    return ::testing::AssertionSuccess();
            }
            const T threshold = std::numeric_limits<T>::min();
            T rel_diff = abs_diff / std::max(threshold,min);
            if (rel_diff > relative_error) {
                std::stringstream message;
                message << "ABS difference at position (" << i << ", " << j << ") is " << std::setprecision(6) << abs_diff
                        << " Rel error is " << rel_diff
                        << ", which exceeds the relative error " << std::setprecision(6) << relative_error
                        << ". Expected: " << std::setprecision(6) << expected[i * cols + j]
                        << ", Actual: " << std::setprecision(6) << actual[i * cols + j];
                std::cerr << message.str();
                return ::testing::AssertionFailure() << message.str();
            }
        }
    }
    return ::testing::AssertionSuccess();
}

#endif // TEST_COMMON_H

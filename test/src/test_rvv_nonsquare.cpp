#include "test_common.hpp"

constexpr size_t nIndex{0U};
constexpr size_t mIndex{1U};
constexpr size_t kIndex{2U};

template <typename T>
class GemmRVVNonSqare : public ::testing::Test
{
public:
    static constexpr size_t n = std::tuple_element_t<nIndex, T>{};
    static constexpr size_t m = std::tuple_element_t<mIndex, T>{};
    static constexpr size_t k = std::tuple_element_t<kIndex, T>{};

    using ElemType = float;
    using VectorType = std::vector<ElemType>;
};

TEST(GemmRVVNonSquare, SimpleTets_1x1) {
    const size_t rows = 1;
    const size_t cols = 1;
    const float A[] = {1.0f};
    const float B[] = {1.0f};
    float C_ref[rows*cols] = {0.0f};
    float C_comp[rows*cols] = {0.0f};

    gemm_ref(A, B, C_ref, rows, cols, rows);
    gemm_block4x4_rvv(A, B, C_comp, rows, cols, rows);

    ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
}

TEST(GemmRVVNonSquare, SimpleTets_2x2) {
    const size_t rows = 2;
    const size_t cols = 2;
    const float A[] = {1.0f, 2.0f,
                       3.0f, 4.0f};
    const float B[] = {1.0f, 2.0f,
                       3.0f, 4.0f};
    float C_ref[rows*cols] = {0.0f};
    float C_comp[rows*cols] = {0.0f};

    gemm_ref(A, B, C_ref, rows, cols, rows);
    gemm_block4x4_rvv(A, B, C_comp, rows, cols, rows);

    ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
}

TEST(GemmRVVNonSquare, SimpleTets_3x3) {
    const size_t rows = 3;
    const size_t cols = 3;
    const float A[] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f};
    const float B[] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f};
    float C_ref[rows*cols] = {0.0f};
    float C_comp[rows*cols] = {0.0f};

    gemm_ref(A, B, C_ref, rows, cols, rows);
    gemm_block4x4_rvv(A, B, C_comp, rows, cols, rows);

    ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
}

#define TEST_GEMM(n, m, k) \
    std::tuple<std::integral_constant<size_t, (n)>, std::integral_constant<size_t, (m)>, std::integral_constant<size_t, (k)>>

using TypesNonSqare = testing::Types<
                                     // Test M
                                     TEST_GEMM(4U, 1U, 4U),
                                     TEST_GEMM(4U, 2U, 4U),
                                     TEST_GEMM(4U, 3U, 4U),
                                     TEST_GEMM(4U, 5U, 4U),
                                     TEST_GEMM(4U, 6U, 4U),
                                     TEST_GEMM(4U, 7U, 4U),
                                     // Test K
                                     TEST_GEMM(4U, 4U, 1U),
                                     TEST_GEMM(4U, 4U, 2U),
                                     TEST_GEMM(4U, 4U, 3U),
                                     TEST_GEMM(4U, 4U, 5U),
                                     TEST_GEMM(4U, 4U, 6U),
                                     TEST_GEMM(4U, 4U, 7U),
                                     // Test M and K
                                     TEST_GEMM(4U, 1U, 1U),
                                     TEST_GEMM(4U, 1U, 2U),
                                     TEST_GEMM(4U, 1U, 3U),
                                     TEST_GEMM(4U, 1U, 5U),
                                     TEST_GEMM(4U, 1U, 6U),
                                     TEST_GEMM(4U, 1U, 7U),
                                     TEST_GEMM(4U, 5U, 1U),
                                     TEST_GEMM(4U, 5U, 2U),
                                     TEST_GEMM(4U, 5U, 3U),
                                     TEST_GEMM(4U, 5U, 5U),
                                     TEST_GEMM(4U, 5U, 6U),
                                     TEST_GEMM(4U, 5U, 7U),
                                     TEST_GEMM(4U, 5U, 5U),
                                     TEST_GEMM(4U, 6U, 5U),
                                     TEST_GEMM(4U, 7U, 5U),
                                     // Test N
                                     TEST_GEMM(1U, 4U, 4U),
                                     TEST_GEMM(2U, 4U, 4U),
                                     TEST_GEMM(3U, 4U, 4U),
                                     TEST_GEMM(5U, 4U, 4U),
                                     TEST_GEMM(6U, 4U, 4U),
                                     TEST_GEMM(7U, 4U, 4U),
                                     // Test N and M
                                     TEST_GEMM(1U, 1U, 4U),
                                     TEST_GEMM(2U, 1U, 4U),
                                     TEST_GEMM(3U, 1U, 4U),
                                     TEST_GEMM(5U, 1U, 4U),
                                     TEST_GEMM(6U, 1U, 4U),
                                     TEST_GEMM(7U, 1U, 4U),
                                     TEST_GEMM(1U, 5U, 4U),
                                     TEST_GEMM(2U, 5U, 4U),
                                     TEST_GEMM(3U, 5U, 4U),
                                     TEST_GEMM(5U, 5U, 4U),
                                     TEST_GEMM(6U, 5U, 4U),
                                     TEST_GEMM(7U, 5U, 4U),
                                     TEST_GEMM(5U, 5U, 4U),
                                     TEST_GEMM(5U, 6U, 4U),
                                     TEST_GEMM(5U, 7U, 4U),
                                     // Test N and K
                                     TEST_GEMM(1U, 4U, 1U),
                                     TEST_GEMM(2U, 4U, 1U),
                                     TEST_GEMM(3U, 4U, 1U),
                                     TEST_GEMM(5U, 4U, 1U),
                                     TEST_GEMM(6U, 4U, 1U),
                                     TEST_GEMM(7U, 4U, 1U),
                                     TEST_GEMM(1U, 4U, 5U),
                                     TEST_GEMM(2U, 4U, 5U),
                                     TEST_GEMM(3U, 4U, 5U),
                                     TEST_GEMM(5U, 4U, 5U),
                                     TEST_GEMM(6U, 4U, 5U),
                                     TEST_GEMM(7U, 4U, 5U),
                                     TEST_GEMM(5U, 4U, 5U),
                                     TEST_GEMM(5U, 4U, 6U),
                                     TEST_GEMM(5U, 4U, 7U),
                                     // Test N M K
                                     TEST_GEMM(1U, 1U, 1U),
                                     TEST_GEMM(2U, 1U, 1U),
                                     TEST_GEMM(3U, 1U, 1U),
                                     TEST_GEMM(5U, 1U, 1U),
                                     TEST_GEMM(6U, 1U, 1U),
                                     TEST_GEMM(7U, 1U, 1U),
                                     TEST_GEMM(1U, 5U, 1U),
                                     TEST_GEMM(2U, 5U, 1U),
                                     TEST_GEMM(3U, 5U, 1U),
                                     TEST_GEMM(5U, 5U, 1U),
                                     TEST_GEMM(6U, 5U, 1U),
                                     TEST_GEMM(7U, 5U, 1U),
                                     TEST_GEMM(1U, 1U, 5U),
                                     TEST_GEMM(2U, 1U, 5U),
                                     TEST_GEMM(3U, 1U, 5U),
                                     TEST_GEMM(5U, 1U, 5U),
                                     TEST_GEMM(6U, 1U, 5U),
                                     TEST_GEMM(7U, 1U, 5U),
                                     TEST_GEMM(1U, 5U, 5U),
                                     TEST_GEMM(2U, 5U, 5U),
                                     TEST_GEMM(3U, 5U, 5U),
                                     TEST_GEMM(5U, 5U, 5U),
                                     TEST_GEMM(6U, 5U, 5U),
                                     TEST_GEMM(7U, 5U, 5U),
                                     TEST_GEMM(5U, 5U, 5U),
                                     TEST_GEMM(5U, 6U, 5U),
                                     TEST_GEMM(5U, 7U, 5U)>;

TYPED_TEST_CASE(GemmRVVNonSqare, TypesNonSqare);

TYPED_TEST(GemmRVVNonSqare, Zero_ABC) 
{
    using ElemType   = typename TestFixture::ElemType;
    using VectorType = typename TestFixture::VectorType;

    const size_t n = TestFixture::n;
    const size_t m = TestFixture::m;
    const size_t k = TestFixture::k;

    const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

    VectorType A(n*m, 0.0f);
    VectorType B(m*k, 0.0f);
    VectorType C_ref(n*k, 0.0f);
    VectorType C_comp(n*k, 0.0f);

    gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
    gemm_block4x4_rvv(A.data(), B.data(), C_comp.data(), n, m, k);

    ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
}

TYPED_TEST(GemmRVVNonSqare, Rand_AB_Zero_C) 
{
    using ElemType   = typename TestFixture::ElemType;
    using VectorType = typename TestFixture::VectorType;

    const size_t n = TestFixture::n;
    const size_t m = TestFixture::m;
    const size_t k = TestFixture::k;

    const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

    VectorType A(n*m, 0.0f);
    VectorType B(m*k, 0.0f);
    VectorType C_ref(n*k, 0.0f);
    VectorType C_comp(n*k, 0.0f);

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<ElemType> dist(0, 100);

    std::generate(A.begin(), A.end(), [&] { return dist(rng); });
    std::generate(B.begin(), B.end(), [&] { return dist(rng); });

    gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
    gemm_block4x4_rvv(A.data(), B.data(), C_comp.data(), n, m, k);

    ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
}

TYPED_TEST(GemmRVVNonSqare, Rand_ABC) 
{
    using ElemType   = typename TestFixture::ElemType;
    using VectorType = typename TestFixture::VectorType;

    const size_t n = TestFixture::n;
    const size_t m = TestFixture::m;
    const size_t k = TestFixture::k;

    const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

    VectorType A(n*m, 0.0f);
    VectorType B(m*k, 0.0f);
    VectorType C_ref(n*k, 0.0f);
    VectorType C_comp(n*k, 0.0f);

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<ElemType> dist(0, 100);

    std::generate(A.begin(), A.end(), [&] { return dist(rng); });
    std::generate(B.begin(), B.end(), [&] { return dist(rng); });
    std::generate(C_ref.begin(), C_ref.end(), [&] { return dist(rng); });
    std::copy(C_ref.begin(), C_ref.end(), C_comp.begin());

    gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
    gemm_block4x4_rvv(A.data(), B.data(), C_comp.data(), n, m, k);

    ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
}

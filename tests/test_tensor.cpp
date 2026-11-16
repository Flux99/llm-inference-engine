#include "test_harness.h"
#include "engine/core/tensor.h"
#include "engine/core/tensor_shape.h"
#include "engine/core/tensor_view.h"
#include "engine/core/dtype.h"
#include "engine/ops/ops_basic.h"

using namespace engine::core;
using namespace engine::ops;

TEST(TensorShapeTest, StridesCalculation) {
    // 1D: [10] -> strides: [1]
    TensorShape shape_1d({10});
    EXPECT_EQ(shape_1d.ndim(), 1);
    EXPECT_EQ(shape_1d.numel(), 10);
    EXPECT_EQ(shape_1d.stride(0), 1);

    // 2D: [2, 3] -> strides: [3, 1]
    TensorShape shape_2d({2, 3});
    EXPECT_EQ(shape_2d.ndim(), 2);
    EXPECT_EQ(shape_2d.numel(), 6);
    EXPECT_EQ(shape_2d.stride(0), 3);
    EXPECT_EQ(shape_2d.stride(1), 1);

    // 3D: [2, 4, 8] -> strides: [32, 8, 1]
    TensorShape shape_3d({2, 4, 8});
    EXPECT_EQ(shape_3d.ndim(), 3);
    EXPECT_EQ(shape_3d.numel(), 64);
    EXPECT_EQ(shape_3d.stride(0), 32);
    EXPECT_EQ(shape_3d.stride(1), 8);
    EXPECT_EQ(shape_3d.stride(2), 1);

    // 4D (e.g. LLM Batch, Head, Seq, Dim): [1, 8, 32, 64]
    TensorShape shape_4d({1, 8, 32, 64});
    EXPECT_EQ(shape_4d.ndim(), 4);
    EXPECT_EQ(shape_4d.numel(), 16384);
    EXPECT_EQ(shape_4d.stride(0), 16384);
    EXPECT_EQ(shape_4d.stride(1), 2048);
    EXPECT_EQ(shape_4d.stride(2), 64);
    EXPECT_EQ(shape_4d.stride(3), 1);
}

TEST(TensorShapeTest, OffsetCalculation) {
    // 2D shape [2, 3] -> offsets: (r, c) = r * 3 + c
    TensorShape shape({2, 3});
    EXPECT_EQ(shape.offset({0, 0}), 0);
    EXPECT_EQ(shape.offset({0, 2}), 2);
    EXPECT_EQ(shape.offset({1, 0}), 3);
    EXPECT_EQ(shape.offset({1, 1}), 4);
    EXPECT_EQ(shape.offset({1, 2}), 5);

    EXPECT_THROW(shape.offset({2, 0}), std::out_of_range);
    EXPECT_THROW(shape.offset({0, 3}), std::out_of_range);
}

TEST(TensorTest, BasicTensorOperations) {
    auto t = Tensor<float>::from_data(TensorShape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    EXPECT_EQ(t(0, 0), 1.0f);
    EXPECT_EQ(t(0, 2), 3.0f);
    EXPECT_EQ(t(1, 1), 5.0f);
    EXPECT_EQ(t(1, 2), 6.0f);

    // Mutate
    t(1, 1) = 99.0f;
    EXPECT_EQ(t(1, 1), 99.0f);
}

TEST(TensorViewTest, ZeroCopySlicing) {
    auto t = Tensor<float>::from_data(TensorShape({4, 2}), {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    });

    // Slice rows [1, 3) -> shape [2, 2], values: [[3, 4], [5, 6]]
    TensorView<float> view = t.view();
    TensorView<float> slice = view.slice_dim0(1, 3);

    EXPECT_EQ(slice.dim(0), 2);
    EXPECT_EQ(slice.dim(1), 2);
    EXPECT_EQ(slice(0, 0), 3.0f);
    EXPECT_EQ(slice(0, 1), 4.0f);
    EXPECT_EQ(slice(1, 0), 5.0f);
    EXPECT_EQ(slice(1, 1), 6.0f);

    // Mutating the slice MUST mutate the underlying tensor (zero-copy proof!)
    slice(0, 0) = 42.0f;
    EXPECT_EQ(t(1, 0), 42.0f);
}

TEST(OpsTest, LinearLayerGroundTruth) {
    // W = [[2, 3], [4, 5]]
    auto W = Tensor<float>::from_data(TensorShape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
    // x = [10, 20]
    auto x = Tensor<float>::from_data(TensorShape({2}), {10.0f, 20.0f});
    // b = [1, 2]
    auto b = Tensor<float>::from_data(TensorShape({2}), {1.0f, 2.0f});

    auto y = Tensor<float>::zeros(TensorShape({2}));

    linear(W.view(), x.view(), b.view(), y.view());

    // y0 = 2*10 + 3*20 + 1 = 81
    // y1 = 4*10 + 5*20 + 2 = 142
    EXPECT_FLOAT_EQ(y(0), 81.0f);
    EXPECT_FLOAT_EQ(y(1), 142.0f);
}

TEST(OpsTest, MatMul2D) {
    // A: 2x3, B: 3x2 -> C: 2x2
    auto A = Tensor<float>::from_data(TensorShape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    auto B = Tensor<float>::from_data(TensorShape({3, 2}), {
        7.0f,  8.0f,
        9.0f,  1.0f,
        2.0f,  3.0f
    });
    auto C = Tensor<float>::zeros(TensorShape({2, 2}));

    matmul(A.view(), B.view(), C.view());

    // C[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    EXPECT_FLOAT_EQ(C(0, 0), 31.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 19.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 85.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 55.0f);
}

TEST(OpsTest, ReLUEvaluation) {
    auto input = Tensor<float>::from_data(TensorShape({4}), {-2.5f, 0.0f, 1.5f, -0.1f});
    auto output = Tensor<float>::zeros(TensorShape({4}));

    relu(input.view(), output.view());

    EXPECT_FLOAT_EQ(output(0), 0.0f);
    EXPECT_FLOAT_EQ(output(1), 0.0f);
    EXPECT_FLOAT_EQ(output(2), 1.5f);
    EXPECT_FLOAT_EQ(output(3), 0.0f);
}

int main() {
    return test_harness::run_all_tests();
}

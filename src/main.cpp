#include <iostream>
#include "engine/core/tensor.h"
#include "engine/core/tensor_shape.h"
#include "engine/core/dtype.h"
#include "engine/ops/ops_basic.h"

using namespace engine::core;
using namespace engine::ops;

int main() {
    std::cout << "======================================================\n";
    std::cout << "   LLM Inference Engine: Phase 1 Tensor Demo\n";
    std::cout << "======================================================\n\n";

    // 1. Inspect 2D Tensor memory layout and strides
    TensorShape shape_2d({2, 3});
    auto matrix = Tensor<float>::from_data(shape_2d, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    std::cout << "1. [2D Matrix] Shape: " << matrix.shape().to_string() << "\n";
    std::cout << "   Strides: [" << matrix.stride(0) << ", " << matrix.stride(1) << "]\n";
    std::cout << "   Total Elements: " << matrix.numel() << "\n";
    std::cout << "   2D Matrix Display:\n";
    matrix.print();
    std::cout << "\n";

    // 2. Demonstration of Zero-Copy Slicing
    std::cout << "2. [Zero-Copy View Slicing]\n";
    auto view = matrix.view();
    auto row_1_slice = view.slice_dim0(1, 2); // Row 1 only
    std::cout << "   Slice Shape: " << row_1_slice.shape().to_string() << "\n";
    std::cout << "   Slice Values:\n";
    row_1_slice.print();
    std::cout << "\n";

    // 3. Execution of Linear Layer: y = W * x + b
    std::cout << "3. [Linear Projection Layer: y = Wx + b]\n";
    auto W = Tensor<float>::from_data(TensorShape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
    auto x = Tensor<float>::from_data(TensorShape({2}), {10.0f, 20.0f});
    auto b = Tensor<float>::from_data(TensorShape({2}), {1.0f, 2.0f});
    auto y = Tensor<float>::zeros(TensorShape({2}));

    std::cout << "   Weight Matrix W (2x2):\n";
    W.print();
    std::cout << "   Input Vector x (2):\n";
    x.print();
    std::cout << "   Bias Vector b (2):\n";
    b.print();

    linear(W.view(), x.view(), b.view(), y.view());

    std::cout << "   Calculated Output Vector y (2):\n";
    y.print();
    std::cout << "   -> Ground truth check: y[0] = 2*10 + 3*20 + 1 = 81\n";
    std::cout << "   -> Ground truth check: y[1] = 4*10 + 5*20 + 2 = 142\n\n";

    std::cout << "✅ Phase 1 Tensor Foundations successfully verified!\n";
    return 0;
}


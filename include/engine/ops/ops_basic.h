#pragma once

#include "engine/core/tensor.h"
#include "engine/core/tensor_view.h"
#include <algorithm>
#include <stdexcept>

namespace engine::ops {

using namespace engine::core;

// Elementwise Add: C = A + B
inline void add(TensorView<const float> a,
                TensorView<const float> b,
                TensorView<float> out) {
    if (a.shape() != b.shape() || a.shape() != out.shape()) {
        throw std::invalid_argument("Shape mismatch in add operation");
    }
    for (size_t i = 0; i < a.numel(); ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
}

// Elementwise ReLU: y = max(0, x)
inline void relu(TensorView<const float> in,
                 TensorView<float> out) {
    if (in.shape() != out.shape()) {
        throw std::invalid_argument("Shape mismatch in relu operation");
    }
    for (size_t i = 0; i < in.numel(); ++i) {
        out.data()[i] = std::max(0.0f, in.data()[i]);
    }
}

// 2D Matrix Multiplication: C(M, N) = A(M, K) * B(K, N)
inline void matmul(TensorView<const float> A,
                   TensorView<const float> B,
                   TensorView<float> C) {
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }
    size_t M = A.dim(0);
    size_t K = A.dim(1);
    size_t K2 = B.dim(0);
    size_t N = B.dim(1);

    if (K != K2 || C.dim(0) != M || C.dim(1) != N) {
        throw std::invalid_argument("Matrix dimension mismatch in matmul");
    }

    // Initialize C to 0
    std::fill(C.data(), C.data() + C.numel(), 0.0f);

    // Standard baseline loop
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = A(i, k);
            for (size_t j = 0; j < N; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
}

// Linear Layer / Projection: y = W * x + b
// W: (M, K) [Weights], x: (K) [Input Vector], b: (M) [Bias], y: (M) [Output Vector]
inline void linear(TensorView<const float> W,
                   TensorView<const float> x,
                   TensorView<const float> b,
                   TensorView<float> y) {
    if (W.ndim() != 2 || x.ndim() != 1 || b.ndim() != 1 || y.ndim() != 1) {
        throw std::invalid_argument("Dimension mismatch in linear layer");
    }
    size_t M = W.dim(0);
    size_t K = W.dim(1);

    if (x.dim(0) != K || b.dim(0) != M || y.dim(0) != M) {
        throw std::invalid_argument("Shape mismatch in linear layer parameters");
    }

    for (size_t i = 0; i < M; ++i) {
        float sum = b(i);
        for (size_t j = 0; j < K; ++j) {
            sum += W(i, j) * x(j);
        }
        y(i) = sum;
    }
}

} // namespace engine::ops

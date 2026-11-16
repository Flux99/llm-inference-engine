#pragma once

#include "engine/core/tensor_shape.h"
#include "engine/core/tensor_view.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>

namespace engine::core {

template <typename T>
class Tensor {
public:
    Tensor() : shape_({0}) {}

    explicit Tensor(TensorShape shape)
        : shape_(std::move(shape)), data_(shape_.numel(), T(0)) {}

    Tensor(TensorShape shape, T fill_value)
        : shape_(std::move(shape)), data_(shape_.numel(), fill_value) {}

    Tensor(TensorShape shape, std::vector<T> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (data_.size() != shape_.numel()) {
            throw std::invalid_argument("Provided data vector size does not match shape numel");
        }
    }

    // Factory methods
    static Tensor<T> zeros(TensorShape shape) {
        return Tensor<T>(shape, T(0));
    }

    static Tensor<T> ones(TensorShape shape) {
        return Tensor<T>(shape, T(1));
    }

    static Tensor<T> from_data(TensorShape shape, std::vector<T> data) {
        return Tensor<T>(std::move(shape), std::move(data));
    }

    // Accessors
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    const TensorShape& shape() const { return shape_; }
    size_t numel() const { return shape_.numel(); }
    size_t ndim() const { return shape_.ndim(); }
    size_t dim(size_t i) const { return shape_.dim(i); }
    size_t stride(size_t i) const { return shape_.stride(i); }

    // Fast 1D access
    T& operator()(size_t i) {
        return data_[i * shape_.stride(0)];
    }
    const T& operator()(size_t i) const {
        return data_[i * shape_.stride(0)];
    }

    // Fast 2D access: row r, col c
    T& operator()(size_t r, size_t c) {
        return data_[r * shape_.stride(0) + c * shape_.stride(1)];
    }
    const T& operator()(size_t r, size_t c) const {
        return data_[r * shape_.stride(0) + c * shape_.stride(1)];
    }

    // General N-D access
    T& operator()(std::initializer_list<size_t> indices) {
        return data_[shape_.offset(indices)];
    }
    const T& operator()(std::initializer_list<size_t> indices) const {
        return data_[shape_.offset(indices)];
    }

    // Implicit conversion to TensorView for zero-copy operator passing
    operator TensorView<T>() {
        return TensorView<T>(data_.data(), shape_);
    }

    operator TensorView<const T>() const {
        return TensorView<const T>(data_.data(), shape_);
    }

    TensorView<T> view() {
        return TensorView<T>(data_.data(), shape_);
    }

    TensorView<const T> view() const {
        return TensorView<const T>(data_.data(), shape_);
    }

    // Reshape (must preserve total element count)
    void reshape(TensorShape new_shape) {
        if (new_shape.numel() != shape_.numel()) {
            throw std::invalid_argument("Reshape element count mismatch");
        }
        shape_ = std::move(new_shape);
    }

    void print() const {
        view().print();
    }

private:
    TensorShape shape_;
    std::vector<T> data_;
};

} // namespace engine::core


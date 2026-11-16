#pragma once

#include "engine/core/tensor_shape.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <type_traits>

namespace engine::core {

template <typename T>
class TensorView {
public:
    TensorView(T* data, TensorShape shape)
        : data_(data), shape_(std::move(shape)) {}

    // Allow implicit conversion from TensorView<U> to TensorView<const U>
    template <typename U>
        requires (std::is_same_v<const U, T>)
    TensorView(const TensorView<U>& other)
        : data_(other.data()), shape_(other.shape()) {}

    T* data() { return data_; }
    const T* data() const { return data_; }

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

    // Sub-view: slice along outer dimension (dim 0) without copying
    TensorView<T> slice_dim0(size_t start, size_t end) const {
        if (start >= end || end > shape_.dim(0)) {
            throw std::out_of_range("Invalid slice bounds");
        }
        std::vector<size_t> new_dims = shape_.dims();
        new_dims[0] = end - start;

        T* new_data = data_ + (start * shape_.stride(0));
        return TensorView<T>(new_data, TensorShape(new_dims));
    }

    void print() const {
        if (ndim() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < dim(0); ++i) {
                std::cout << (*this)(i) << (i + 1 < dim(0) ? ", " : "");
            }
            std::cout << "]\n";
        } else if (ndim() == 2) {
            std::cout << "[\n";
            for (size_t r = 0; r < dim(0); ++r) {
                std::cout << "  [";
                for (size_t c = 0; c < dim(1); ++c) {
                    std::cout << std::setw(6) << std::setprecision(4) << (*this)(r, c)
                              << (c + 1 < dim(1) ? ", " : "");
                }
                std::cout << "]" << (r + 1 < dim(0) ? ",\n" : "\n");
            }
            std::cout << "]\n";
        } else {
            std::cout << "TensorView(shape=" << shape_.to_string() << ", data_ptr=" << data_ << ")\n";
        }
    }

private:
    T* data_{nullptr};
    TensorShape shape_;
};

} // namespace engine::core

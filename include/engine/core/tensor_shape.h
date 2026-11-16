#pragma once // Prevent duplicate inclusion

#include <vector>           // Dynamic array to store arbitrary dimension counts
#include <initializer_list> // Allows syntax: TensorShape({2, 3})
#include <span>             // C++20 non-owning contiguous view of indices
#include <string>           // For error messages and to_string()
#include <sstream>          // For building formatted string representations
#include <numeric>          // For std::accumulate (optional math utilities)
#include <stdexcept>        // For std::out_of_range and std::invalid_argument

namespace engine::core {

class TensorShape {
public:
    // Default constructor: creates an empty scalar/0-dim shape
    TensorShape() : dims_({0}), strides_({1}), numel_(0) {}

    // Constructor from an initializer list, e.g. TensorShape({2, 3, 4})
    TensorShape(std::initializer_list<size_t> dims)
        : dims_(dims) {
        compute_strides_and_numel();
    }

    // Constructor from an existing vector of dimensions
    explicit TensorShape(std::vector<size_t> dims)
        : dims_(std::move(dims)) {
        compute_strides_and_numel();
    }

    // Returns number of dimensions (rank of the tensor)
    size_t ndim() const { return dims_.size(); }

    // Returns the size of dimension i with bounds check
    size_t dim(size_t i) const { return dims_.at(i); }

    // Returns a const reference to all dimensions vector
    const std::vector<size_t>& dims() const { return dims_; }

    // Returns stride for dimension i with bounds check
    size_t stride(size_t i) const { return strides_.at(i); }

    // Returns a const reference to all strides vector
    const std::vector<size_t>& strides() const { return strides_; }

    // Returns total number of elements across all dimensions
    size_t numel() const { return numel_; }

    // Translates multi-dimensional coordinate {i0, i1, ...} into a single 1D flat memory index
    size_t offset(std::initializer_list<size_t> indices) const {
        if (indices.size() != dims_.size()) {
            throw std::invalid_argument("Index dimension count mismatch");
        }
        size_t off = 0;
        size_t d = 0;
        for (size_t idx : indices) {
            if (idx >= dims_[d]) {
                throw std::out_of_range("Index out of bounds at dimension " + std::to_string(d));
            }
            off += idx * strides_[d];
            d++;
        }
        return off;
    }

    // Overload for std::span to avoid copies when coordinates are already in a container
    size_t offset(std::span<const size_t> indices) const {
        if (indices.size() != dims_.size()) {
            throw std::invalid_argument("Index dimension count mismatch");
        }
        size_t off = 0;
        for (size_t d = 0; d < indices.size(); ++d) {
            if (indices[d] >= dims_[d]) {
                throw std::out_of_range("Index out of bounds at dimension " + std::to_string(d));
            }
            off += indices[d] * strides_[d];
        }
        return off;
    }

    // Equality operator
    bool operator==(const TensorShape& other) const {
        return dims_ == other.dims_;
    }

    // Inequality operator
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }

    // Pretty-prints shape as a string: "[2, 3]"
    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims_.size(); ++i) {
            oss << dims_[i];
            if (i + 1 < dims_.size()) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

private:
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
    size_t numel_{0};

    // Computes row-major contiguous strides from right to left:
    // stride[last] = 1
    // stride[k] = stride[k+1] * dims[k+1]
    void compute_strides_and_numel() {
        if (dims_.empty()) {
            numel_ = 0;
            return;
        }

        strides_.resize(dims_.size());
        numel_ = 1;
        for (size_t d : dims_) {
            numel_ *= d;
        }

        if (!dims_.empty()) {
            strides_.back() = 1;
            for (int k = static_cast<int>(dims_.size()) - 2; k >= 0; --k) {
                strides_[k] = strides_[k + 1] * dims_[k + 1];
            }
        }
    }
};

} // namespace engine::core

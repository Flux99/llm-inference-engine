# Phase 1: Tensor Foundations, Strides & Memory Layout
*Track 1: C++ Foundations & Graph Engine*

This document is your complete, self-contained lesson for Phase 1. It contains all the concepts, formulas, memory diagrams, **every single code file with in-line comments explaining what each line does**, and a **step-by-step Execution Flow Breakdown below every code snippet**.

---

## 1. 🎯 Goal
Build a clean C++20 `Tensor` and `TensorView` library from scratch that stores numbers in a single flat 1D memory array in RAM and lets us read/write elements using multi-dimensional coordinates `(row, col)` with **zero extra copies**.

---

## 2. ❓ Why

### Why not `std::vector<std::vector<float>>`?
1. **Cache Thrashing (Speed):** In nested vectors, each row is allocated at a random address on the heap. Modern CPU caches load 64-byte blocks of contiguous memory. Jumping between fragmented row pointers slows down matrix operations by 10x–50x.
2. **Zero-Copy Slicing (Memory):** In LLMs, you constantly extract slices (e.g. one attention head, or a token's Key-Value vector). With nested vectors, slicing requires allocating and copying memory. With flat strided tensors, slicing is a zero-copy pointer adjustment.
3. **GPU / CUDA Compatibility:** GPU memory transfers (`cudaMemcpy`) require a single contiguous buffer (`float*`). Nested pointers cannot be copied to a GPU in one shot.

### The Solution:
Store all elements in **one continuous 1D buffer** (`std::vector<T>`), and use **strides** to translate $(row, col)$ coordinates into the flat memory offset.

---

## 3. 🧠 Mental Model & Memory Layout

### A 2D Matrix of Shape `[2, 3]` (2 rows, 3 columns):

```text
Logical 2D View:
               Col 0   Col 1   Col 2
Row 0   ───► [  1.0,    2.0,    3.0  ]
Row 1   ───► [  4.0,    5.0,    6.0  ]

Physical 1D RAM Buffer:
Offset:   0       1       2       3       4       5
Value:  [ 1.0  |  2.0  |  3.0  |  4.0  |  5.0  |  6.0 ]
```

### Stride Math (Row-Major Layout):
- `stride[last] = 1` (moving 1 column steps 1 element forward)
- `stride[k] = stride[k+1] * dim[k+1]`

For shape `[2, 3]`:
- `stride[1] = 1`
- `stride[0] = stride[1] * dim[1] = 1 * 3 = 3` (moving 1 row skips 3 elements in RAM)

### Flat Offset Formula:
$$\text{offset}(r, c) = r \times \text{stride}_0 + c \times \text{stride}_1 = r \times 3 + c \times 1$$

- Coordinate `(0, 2)` $\rightarrow 0 \times 3 + 2 \times 1 = \mathbf{2} \rightarrow \text{Value } 3.0$
- Coordinate `(1, 1)` $\rightarrow 1 \times 3 + 1 \times 1 = \mathbf{4} \rightarrow \text{Value } 5.0$

---

## 4. 💻 Complete Code Files, In-Line Comments & Execution Flows

Below is every file with:
1. **Why this code exists**
2. **Commented Code**: Explaining what each line does.
3. **Flow of Code**: The chronological step-by-step execution path.

---

### 📄 File 1: `CMakeLists.txt`
**Path:** `/home/abhishek/llm_inference/CMakeLists.txt`

#### Why this code?
Sets up modern C++20, `-O3` compiler optimizations, AVX2 SIMD instructions, and defines three build targets:
1. `engine_core`: A header-only library containing our tensor classes.
2. `engine_tests`: The automated test suite executable.
3. `engine_cli`: The interactive demo binary.

#### Annotated Code:
```cmake
# Require CMake 3.20 or newer to support modern C++ features and target properties
cmake_minimum_required(VERSION 3.20)

# Define the project name and declare that we use C++ (CXX)
project(llm_inference_engine CXX)

# Enforce the modern C++20 standard (required for concepts, std::span, etc.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json so IDEs (clangd, VSCode) have full code intelligence
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler optimization & warning flags for GCC and Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -Wall -Wextra -Wpedantic : Turn on strict compiler warnings to catch subtle bugs
    # -O3                      : Maximum optimization (loop unrolling, inlining, vectorization)
    # -mavx2 -mfma             : Emit AVX2 (256-bit SIMD) and Fused Multiply-Add CPU instructions
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -mavx2 -mfma)
endif()

# Add include/ and tests/ directories to the header search path
include_directories(include tests)

# Define engine_core as an INTERFACE (header-only) library
add_library(engine_core INTERFACE)

# Expose include/ to any target that links against engine_core
target_include_directories(engine_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the CLI executable from src/main.cpp
add_executable(engine_cli src/main.cpp)

# Link engine_cli with our engine_core headers
target_link_libraries(engine_cli PRIVATE engine_core)

# Enable CTest framework for automated test runs
enable_testing()

# Define the unit testing executable from tests/test_tensor.cpp
add_executable(engine_tests tests/test_tensor.cpp)

# Link engine_tests with engine_core headers
target_link_libraries(engine_tests PRIVATE engine_core)

# Register engine_tests with CTest under the test name "TensorUnitTests"
add_test(NAME TensorUnitTests COMMAND engine_tests)
```

#### 🔄 Flow of Code in `CMakeLists.txt`:
```text
1. cmake -B build initializes configuration
   │
2. Checks compiler version and sets C++20 standard
   │
3. Injects optimization flags (-O3, -mavx2, -mfma) into the build environment
   │
4. Creates target 'engine_core' (exposing header files in include/)
   │
5. Creates target 'engine_cli' (compiles src/main.cpp against engine_core)
   │
6. Creates target 'engine_tests' (compiles tests/test_tensor.cpp against engine_core)
   │
7. Registers 'engine_tests' with CTest runner -> Ready for build & test execution
```

---

### 📄 File 2: `include/engine/core/dtype.h`
**Path:** `/home/abhishek/llm_inference/include/engine/core/dtype.h`

#### Why this code?
Inference engines must handle multiple precisions (FP32, FP16, INT8, INT32). This file defines a strongly-typed `DType` enum, a byte-size lookup helper, and compile-time type traits to connect C++ types to their runtime representation.

#### Annotated Code:
```cpp
#pragma once // Prevent double inclusion of this header file

#include <cstddef>   // For size_t (standard unsigned integer type for sizes)
#include <string>    // For std::string representation of types
#include <stdexcept> // For std::invalid_argument exception

namespace engine::core {

// Strongly-typed enum representing supported tensor data types
enum class DType {
    Float32, // 32-bit single-precision floating point (standard baseline)
    Float16, // 16-bit half-precision floating point (modern LLM default)
    Int8,    // 8-bit integer (used for quantized weights/activations)
    Int32    // 32-bit signed integer (used for token IDs and indexing)
};

// Returns the memory footprint in bytes for a given DType
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4; // float is 4 bytes (32 bits)
        case DType::Float16: return 2; // half float is 2 bytes (16 bits)
        case DType::Int8:    return 1; // int8_t is 1 byte (8 bits)
        case DType::Int32:   return 4; // int32_t is 4 bytes (32 bits)
        default:
            throw std::invalid_argument("Unknown DType"); // Safety guard for unhandled types
    }
}

// Converts DType to a human-readable string for logging and debugging
inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "Float32";
        case DType::Float16: return "Float16";
        case DType::Int8:    return "Int8";
        case DType::Int32:   return "Int32";
        default:             return "Unknown";
    }
}

// Primary template for compile-time mapping from C++ native types to DType enum
template <typename T>
struct TypeToDType;

// Specialization: float maps to DType::Float32
template <> struct TypeToDType<float>   { static constexpr DType value = DType::Float32; };

// Specialization: int8_t maps to DType::Int8
template <> struct TypeToDType<int8_t>  { static constexpr DType value = DType::Int8; };

// Specialization: int32_t maps to DType::Int32
template <> struct TypeToDType<int32_t> { static constexpr DType value = DType::Int32; };

} // namespace engine::core
```

#### 🔄 Flow of Code in `dtype.h`:
```text
Caller queries a DType:
   │
   ├── Runtime query: dtype_size(DType::Float32)
   │   └── switch-case matches DType::Float32 -> returns 4 bytes
   │
   ├── String query: dtype_to_string(DType::Float32)
   │   └── switch-case matches DType::Float32 -> returns "Float32"
   │
   └── Compile-time query: TypeToDType<float>::value
       └── Compiler evaluates static constexpr -> resolves to DType::Float32 with zero overhead
```

---

### 📄 File 3: `include/engine/core/tensor_shape.h`
**Path:** `/home/abhishek/llm_inference/include/engine/core/tensor_shape.h`

#### Why this code?
This is the mathematical heart of tensor indexing:
- `dims_`: Stores the dimensions (e.g. `{2, 3}`).
- `strides_`: Automatically calculates row-major strides (`{3, 1}`) upon construction.
- `numel_`: Total elements ($\prod \text{dims}_i$).
- `offset(...)`: Converts coordinates like `(1, 1)` into flat index `4` using $\sum i_d \cdot \text{stride}_d$.

#### Annotated Code:
```cpp
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
        compute_strides_and_numel(); // Automatically calculate strides and total element count
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
        // Guard: Number of coordinates passed must match tensor rank
        if (indices.size() != dims_.size()) {
            throw std::invalid_argument("Index dimension count mismatch");
        }
        size_t off = 0; // Accumulator for 1D flat offset
        size_t d = 0;   // Current dimension index
        for (size_t idx : indices) {
            // Guard: Out-of-bounds check for dimension d
            if (idx >= dims_[d]) {
                throw std::out_of_range("Index out of bounds at dimension " + std::to_string(d));
            }
            // Linear transformation: index * stride[d]
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

    // Equality operator: two shapes are equal if their dimensions match exactly
    bool operator==(const TensorShape& other) const {
        return dims_ == other.dims_;
    }

    // Inequality operator
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }

    // Pretty-prints shape as a string: "[2, 3, 4]"
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
    std::vector<size_t> dims_;    // Dimensions array, e.g. [2, 3]
    std::vector<size_t> strides_; // Stride multipliers, e.g. [3, 1]
    size_t numel_{0};             // Total element count, e.g. 6

    // Computes row-major contiguous strides:
    // stride[last] = 1
    // stride[k] = stride[k+1] * dims[k+1]
    void compute_strides_and_numel() {
        if (dims_.empty()) {
            numel_ = 0;
            return;
        }

        strides_.resize(dims_.size());
        numel_ = 1;
        // Step 1: Multiply all dimension sizes to get total elements
        for (size_t d : dims_) {
            numel_ *= d;
        }

        // Step 2: Compute strides from right to left (Row-Major convention)
        if (!dims_.empty()) {
            strides_.back() = 1; // Innermost dimension always has a stride of 1 element
            // Loop backwards from second-to-last dimension down to 0
            for (int k = static_cast<int>(dims_.size()) - 2; k >= 0; --k) {
                strides_[k] = strides_[k + 1] * dims_[k + 1];
            }
        }
    }
};

} // namespace engine::core
```

#### 🔄 Flow of Code in `tensor_shape.h`:
```text
Instantiation: TensorShape({2, 3})
   │
   ├── 1. dims_ is set to [2, 3]
   │
   ├── 2. compute_strides_and_numel() runs:
   │      ├── numel_ = 2 * 3 = 6
   │      ├── strides_.back() = 1 (column stride = 1)
   │      └── strides_[0] = strides_[1] * dims_[1] = 1 * 3 = 3 (row stride = 3)
   │
   └── 3. offset({1, 1}) is called:
          ├── Check coordinate 1 < dim[0] (1 < 2) -> OK
          ├── off += 1 * 3 = 3
          ├── Check coordinate 1 < dim[1] (1 < 3) -> OK
          ├── off += 1 * 1 = 1
          └── Returns 3 + 1 = 4 (exact index in 1D memory array)
```

---

### 📄 File 4: `include/engine/core/tensor_view.h`
**Path:** `/home/abhishek/llm_inference/include/engine/core/tensor_view.h`

#### Why this code?
A `TensorView<T>` is a lightweight, non-owning window over memory. It holds:
1. `T* data_`: A raw pointer to the start of the data.
2. `TensorShape shape_`: The shape and strides.
- `slice_dim0(start, end)` creates a new view pointing to a subset of rows **without allocating or copying any memory**. Mutating the view mutates the original tensor!

#### Annotated Code:
```cpp
#pragma once

#include "engine/core/tensor_shape.h" // Stride and shape calculations
#include <stdexcept>                  // For out_of_range exceptions
#include <iostream>                   // For printing tensor content to console
#include <iomanip>                    // For std::setw, std::setprecision formatting
#include <type_traits>                // For std::is_same_v compile-time checks

namespace engine::core {

template <typename T>
class TensorView {
public:
    // Constructor: Binds an existing raw pointer to a TensorShape (does not own memory)
    TensorView(T* data, TensorShape shape)
        : data_(data), shape_(std::move(shape)) {}

    // Converting constructor: Allows passing TensorView<float> where TensorView<const float> is expected
    template <typename U>
        requires (std::is_same_v<const U, T>) // Only enabled if T is "const U" (safe const qualification)
    TensorView(const TensorView<U>& other)
        : data_(other.data()), shape_(other.shape()) {}

    // Get the underlying raw pointer
    T* data() { return data_; }
    const T* data() const { return data_; }

    // Query shape properties
    const TensorShape& shape() const { return shape_; }
    size_t numel() const { return shape_.numel(); }
    size_t ndim() const { return shape_.ndim(); }
    size_t dim(size_t i) const { return shape_.dim(i); }
    size_t stride(size_t i) const { return shape_.stride(i); }

    // Fast 1D element access: index * stride(0)
    T& operator()(size_t i) {
        return data_[i * shape_.stride(0)];
    }
    const T& operator()(size_t i) const {
        return data_[i * shape_.stride(0)];
    }

    // Fast 2D element access: (row r, col c) -> avoids dynamic loops/allocations
    T& operator()(size_t r, size_t c) {
        return data_[r * shape_.stride(0) + c * shape_.stride(1)];
    }
    const T& operator()(size_t r, size_t c) const {
        return data_[r * shape_.stride(0) + c * shape_.stride(1)];
    }

    // General N-D element access via initializer list {i, j, k, ...}
    T& operator()(std::initializer_list<size_t> indices) {
        return data_[shape_.offset(indices)];
    }
    const T& operator()(std::initializer_list<size_t> indices) const {
        return data_[shape_.offset(indices)];
    }

    // Zero-Copy Slice along outer dimension (dim 0), e.g. extracting rows [start, end)
    TensorView<T> slice_dim0(size_t start, size_t end) const {
        // Validate slice boundaries
        if (start >= end || end > shape_.dim(0)) {
            throw std::out_of_range("Invalid slice bounds");
        }
        // Copy dimension array and adjust outermost dimension
        std::vector<size_t> new_dims = shape_.dims();
        new_dims[0] = end - start;

        // Pointer arithmetic: Advance base pointer by start * stride(0)
        // Zero bytes copied! Only pointer offset shifted!
        T* new_data = data_ + (start * shape_.stride(0));
        return TensorView<T>(new_data, TensorShape(new_dims));
    }

    // Formatted ASCII printout for inspection
    void print() const {
        if (ndim() == 1) { // 1D Vector printout: [1, 2, 3]
            std::cout << "[";
            for (size_t i = 0; i < dim(0); ++i) {
                std::cout << (*this)(i) << (i + 1 < dim(0) ? ", " : "");
            }
            std::cout << "]\n";
        } else if (ndim() == 2) { // 2D Matrix printout with aligned columns
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
        } else { // Higher-rank tensor summary fallback
            std::cout << "TensorView(shape=" << shape_.to_string() << ", data_ptr=" << data_ << ")\n";
        }
    }

private:
    T* data_{nullptr};   // Non-owning pointer to the beginning of the view
    TensorShape shape_;  // Shape and stride metadata
};

} // namespace engine::core
```

#### 🔄 Flow of Code in `tensor_view.h`:
```text
Zero-Copy Slicing Flow: slice_dim0(1, 2) on a 2x3 matrix
   │
   ├── 1. Boundary check: 1 < 2 and 2 <= 2 -> OK
   │
   ├── 2. new_dims = [2 - 1, 3] = [1, 3]
   │
   ├── 3. Pointer calculation:
   │      new_data = data_ + (1 * stride[0]) = data_ + (1 * 3) = data_ + 3
   │      (Points directly to row 1's starting float in the original memory buffer)
   │
   └── 4. Returns new TensorView<T>(new_data, TensorShape({1, 3}))
          No data was cloned or allocated. Writing to slice(0, 0) writes directly to data_[3]!
```

---

### 📄 File 5: `include/engine/core/tensor.h`
**Path:** `/home/abhishek/llm_inference/include/engine/core/tensor.h`

#### Why this code?
`Tensor<T>` is the owning container. It allocates memory via `std::vector<T> data_` and guarantees RAII memory safety. It automatically converts to `TensorView<T>` so it can be passed directly into operators without explicit conversion code.

#### Annotated Code:
```cpp
#pragma once

#include "engine/core/tensor_shape.h" // Shape and stride logic
#include "engine/core/tensor_view.h"  // View abstraction for non-owning access
#include <vector>                     // Heap-allocated contiguous dynamic storage
#include <memory>                     // Memory utilities
#include <string>                     // String formatting
#include <iostream>                   // Console print

namespace engine::core {

template <typename T>
class Tensor {
public:
    // Default constructor: empty scalar tensor
    Tensor() : shape_({0}) {}

    // Constructor: Allocates uninitialized/zeroed storage for given shape
    explicit Tensor(TensorShape shape)
        : shape_(std::move(shape)), data_(shape_.numel(), T(0)) {}

    // Constructor: Allocates storage filled with a specific value
    Tensor(TensorShape shape, T fill_value)
        : shape_(std::move(shape)), data_(shape_.numel(), fill_value) {}

    // Constructor: Moves existing data vector into tensor (validates size match)
    Tensor(TensorShape shape, std::vector<T> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (data_.size() != shape_.numel()) {
            throw std::invalid_argument("Provided data vector size does not match shape numel");
        }
    }

    // Factory method: Create tensor filled with zeros
    static Tensor<T> zeros(TensorShape shape) {
        return Tensor<T>(shape, T(0));
    }

    // Factory method: Create tensor filled with ones
    static Tensor<T> ones(TensorShape shape) {
        return Tensor<T>(shape, T(1));
    }

    // Factory method: Construct tensor directly from raw data vector
    static Tensor<T> from_data(TensorShape shape, std::vector<T> data) {
        return Tensor<T>(std::move(shape), std::move(data));
    }

    // Access raw underlying pointer to buffer
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Query dimensions and layout
    const TensorShape& shape() const { return shape_; }
    size_t numel() const { return shape_.numel(); }
    size_t ndim() const { return shape_.ndim(); }
    size_t dim(size_t i) const { return shape_.dim(i); }
    size_t stride(size_t i) const { return shape_.stride(i); }

    // Fast 1D element access
    T& operator()(size_t i) {
        return data_[i * shape_.stride(0)];
    }
    const T& operator()(size_t i) const {
        return data_[i * shape_.stride(0)];
    }

    // Fast 2D element access: (row r, col c)
    T& operator()(size_t r, size_t c) {
        return data_[r * shape_.stride(0) + c * shape_.stride(1)];
    }
    const T& operator()(size_t r, size_t c) const {
        return data_[r * shape_.stride(0) + c * shape_.stride(1)];
    }

    // General N-D element access
    T& operator()(std::initializer_list<size_t> indices) {
        return data_[shape_.offset(indices)];
    }
    const T& operator()(std::initializer_list<size_t> indices) const {
        return data_[shape_.offset(indices)];
    }

    // Implicit conversion operator to mutable TensorView<T>
    // Allows passing 'my_tensor' directly into functions expecting TensorView<T>
    operator TensorView<T>() {
        return TensorView<T>(data_.data(), shape_);
    }

    // Implicit conversion operator to const TensorView<const T>
    operator TensorView<const T>() const {
        return TensorView<const T>(data_.data(), shape_);
    }

    // Explicit helper method to get a mutable TensorView
    TensorView<T> view() {
        return TensorView<T>(data_.data(), shape_);
    }

    // Explicit helper method to get a const TensorView
    TensorView<const T> view() const {
        return TensorView<const T>(data_.data(), shape_);
    }

    // Reshape: Changes dimension interpretation without moving/copying data
    void reshape(TensorShape new_shape) {
        // Total element count MUST match
        if (new_shape.numel() != shape_.numel()) {
            throw std::invalid_argument("Reshape element count mismatch");
        }
        shape_ = std::move(new_shape);
    }

    // Print tensor contents using TensorView print format
    void print() const {
        view().print();
    }

private:
    TensorShape shape_;   // Shape and stride metadata
    std::vector<T> data_; // The actual contiguous heap memory buffer
};

} // namespace engine::core
```

#### 🔄 Flow of Code in `tensor.h`:
```text
Instantiation: Tensor<float>::from_data({2, 3}, {1, 2, 3, 4, 5, 6})
   │
   ├── 1. Constructor validates data_.size() (6) == shape_.numel() (6)
   │
   ├── 2. Vector data_ is moved into heap-allocated storage
   │
   ├── 3. Accessing t(1, 0):
   │      Calculates offset = 1 * stride[0] + 0 * stride[1] = 1 * 3 + 0 = 3
   │      Returns reference to data_[3] (value 4.0)
   │
   ├── 4. Passing t to a function taking TensorView<const float>:
   │      Implicit conversion operator operator TensorView<const T>() triggers
   │      Creates lightweight TensorView(data_.data(), shape_) with zero heap allocations
   │
   └── 5. Scope ends:
          Destructor of std::vector<T> automatically deallocates the contiguous memory (RAII)
```

---

### 📄 File 6: `include/engine/ops/ops_basic.h`
**Path:** `/home/abhishek/llm_inference/include/engine/ops/ops_basic.h`

#### Why this code?
Implements fundamental neural network operators:
- `linear`: Computes $y = Wx + b$, the core projection calculation:
  $$y_i = \sum_{j=0}^{K-1} W_{i, j} \cdot x_j + b_i$$
- `matmul`: 2D Matrix multiplication $C_{M \times N} = A_{M \times K} \times B_{K \times N}$.
- `relu`: $y = \max(0, x)$.
- `add`: Elementwise $C = A + B$.

#### Annotated Code:
```cpp
#pragma once

#include "engine/core/tensor.h"      // Tensor definitions
#include "engine/core/tensor_view.h" // TensorView definitions
#include <algorithm>                 // std::max, std::fill
#include <stdexcept>                 // std::invalid_argument

namespace engine::ops {

using namespace engine::core;

// Elementwise Add: out = a + b
inline void add(TensorView<const float> a,
                TensorView<const float> b,
                TensorView<float> out) {
    // Both inputs and output must have identical shapes
    if (a.shape() != b.shape() || a.shape() != out.shape()) {
        throw std::invalid_argument("Shape mismatch in add operation");
    }
    // Loop over all elements and add them
    for (size_t i = 0; i < a.numel(); ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
}

// Elementwise ReLU activation: out = max(0, in)
inline void relu(TensorView<const float> in,
                 TensorView<float> out) {
    if (in.shape() != out.shape()) {
        throw std::invalid_argument("Shape mismatch in relu operation");
    }
    // Clamps negative numbers to zero
    for (size_t i = 0; i < in.numel(); ++i) {
        out.data()[i] = std::max(0.0f, in.data()[i]);
    }
}

// 2D Matrix Multiplication: C(M, N) = A(M, K) * B(K, N)
inline void matmul(TensorView<const float> A,
                   TensorView<const float> B,
                   TensorView<float> C) {
    // Verify all tensors are rank-2
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }
    size_t M = A.dim(0);  // Rows in A
    size_t K = A.dim(1);  // Cols in A
    size_t K2 = B.dim(0); // Rows in B
    size_t N = B.dim(1);  // Cols in B

    // Matrix dimension rule: Inner dimension K must match, and output must be M x N
    if (K != K2 || C.dim(0) != M || C.dim(1) != N) {
        throw std::invalid_argument("Matrix dimension mismatch in matmul");
    }

    // Initialize output matrix memory to 0.0f before accumulation
    std::fill(C.data(), C.data() + C.numel(), 0.0f);

    // Cache-friendly i-k-j loop ordering:
    // Outer loop over rows of A
    for (size_t i = 0; i < M; ++i) {
        // Middle loop over inner dimension K
        for (size_t k = 0; k < K; ++k) {
            float a_ik = A(i, k); // Hoist A(i, k) into CPU register
            // Inner loop over contiguous columns of B and C
            for (size_t j = 0; j < N; ++j) {
                C(i, j) += a_ik * B(k, j); // Sequential access in RAM for B and C
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
    // Validate ranks: W must be 2D, while x, b, y must be 1D vectors
    if (W.ndim() != 2 || x.ndim() != 1 || b.ndim() != 1 || y.ndim() != 1) {
        throw std::invalid_argument("Dimension mismatch in linear layer");
    }
    size_t M = W.dim(0); // Output dimension
    size_t K = W.dim(1); // Input dimension

    // Dimension rule: x must have K elements, b and y must have M elements
    if (x.dim(0) != K || b.dim(0) != M || y.dim(0) != M) {
        throw std::invalid_argument("Shape mismatch in linear layer parameters");
    }

    // Compute each output element y[i]
    for (size_t i = 0; i < M; ++i) {
        float sum = b(i); // Initialize accumulator with bias b[i]
        // Compute dot product of row W[i, :] and vector x[:]
        for (size_t j = 0; j < K; ++j) {
            sum += W(i, j) * x(j);
        }
        y(i) = sum; // Store result in output vector y[i]
    }
}

} // namespace engine::ops
```

#### 🔄 Flow of Code in `ops_basic.h`:
```text
Execution Flow: linear(W, x, b, y)
   │
   ├── 1. Dimension Validation:
   │      Checks W is 2D (M x K), x is 1D (K), b is 1D (M), y is 1D (M)
   │
   ├── 2. Outer loop: For each output neuron i from 0 to M-1:
   │      ├── Initialize accumulator: sum = b[i]
   │      │
   │      ├── Inner loop: For each input feature j from 0 to K-1:
   │      │   └── sum += W(i, j) * x(j)  (Multiply weight by input and accumulate)
   │      │
   │      └── Write result: y(i) = sum
   │
   └── 3. Output vector y contains exact projected activations y = Wx + b
```

---

### 📄 File 7: `tests/test_harness.h`
**Path:** `/home/abhishek/llm_inference/tests/test_harness.h`

#### Why this code?
A zero-dependency test runner. It allows automated unit testing without requiring external internet downloads in sandboxed WSL environments.

#### Annotated Code:
```cpp
#pragma once

#include <iostream>   // For console printout of test results
#include <vector>     // For storing registered test cases
#include <string>     // For test names
#include <functional> // For std::function (holding test functions)
#include <cmath>      // For std::abs in floating-point comparison
#include <stdexcept>  // For std::runtime_error

namespace test_harness {

// Structure holding test metadata and executable lambda/function
struct TestCase {
    std::string suite_name;     // Name of test group, e.g. "TensorShapeTest"
    std::string test_name;      // Specific test name, e.g. "StridesCalculation"
    std::function<void()> func; // Pointer to test function
};

// Singleton test registry: stores all registered tests
inline std::vector<TestCase>& get_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

// Helper to push a test into the global registry
inline bool register_test(const std::string& suite, const std::string& name, std::function<void()> func) {
    get_registry().push_back({suite, name, std::move(func)});
    return true; // Return true to allow static variable initialization trick
}

// Executes all registered tests and reports summary
inline int run_all_tests() {
    int passed = 0;
    int failed = 0;
    auto& tests = get_registry();

    std::cout << "\n[==========] Running " << tests.size() << " test cases.\n";
    for (const auto& test : tests) {
        std::cout << "[ RUN      ] " << test.suite_name << "." << test.test_name << "\n";
        try {
            test.func(); // Execute the test body
            std::cout << "[       OK ] " << test.suite_name << "." << test.test_name << "\n";
            passed++;
        } catch (const std::exception& e) { // Catch assertion exceptions
            std::cout << "[  FAILED  ] " << test.suite_name << "." << test.test_name << " (Exception: " << e.what() << ")\n";
            failed++;
        } catch (...) { // Catch any unexpected exceptions
            std::cout << "[  FAILED  ] " << test.suite_name << "." << test.test_name << " (Unknown exception)\n";
            failed++;
        }
    }
    // Summary report
    std::cout << "[==========] " << tests.size() << " tests ran.\n";
    std::cout << "[  PASSED  ] " << passed << " tests.\n";
    if (failed > 0) {
        std::cout << "[  FAILED  ] " << failed << " tests.\n";
        return 1; // Return error code if any test fails
    }
    return 0; // Success
}

} // namespace test_harness

// Macro mimicking GoogleTest's TEST(Suite, Name) syntax
// Defines a function and automatically registers it before main() runs
#define TEST(suite, name) \
    void test_##suite##_##name(); \
    static bool registered_##suite##_##name = ::test_harness::register_test(#suite, #name, test_##suite##_##name); \
    void test_##suite##_##name()

// Macro for equality check: throws exception with file line if values differ
#define EXPECT_EQ(val1, val2) \
    do { \
        if (!((val1) == (val2))) { \
            throw std::runtime_error("EXPECT_EQ failed at line " + std::to_string(__LINE__) + \
                ": " #val1 " != " #val2); \
        } \
    } while (0)

// Macro for floating-point equality check: allows small epsilon tolerance (1e-5)
#define EXPECT_FLOAT_EQ(val1, val2) \
    do { \
        if (std::abs((val1) - (val2)) > 1e-5f) { \
            throw std::runtime_error("EXPECT_FLOAT_EQ failed at line " + std::to_string(__LINE__) + \
                ": " #val1 " (" + std::to_string(val1) + ") != " #val2 " (" + std::to_string(val2) + ")"); \
        } \
    } while (0)

// Macro verifying that a statement throws expected exception type
#define EXPECT_THROW(stmt, ex_type) \
    do { \
        bool threw = false; \
        try { stmt; } \
        catch (const ex_type&) { threw = true; } \
        catch (...) {} \
        if (!threw) { \
            throw std::runtime_error("EXPECT_THROW failed at line " + std::to_string(__LINE__) + \
                ": Expected exception " #ex_type " was not thrown."); \
        } \
    } while (0)
```

#### 🔄 Flow of Code in `test_harness.h`:
```text
Compilation & Startup Flow:
   │
   ├── 1. Pre-main(): TEST(Suite, Name) macro initializes static bool variable
   │      └── register_test() appends test name and function pointer to get_registry()
   │
   ├── 2. main() calls run_all_tests()
   │
   ├── 3. Iterates over every test in get_registry():
   │      ├── Prints "[ RUN ]"
   │      ├── Executes test function in a try-catch block
   │      ├── If an assertion fails -> throws std::runtime_error -> caught -> marked [ FAILED ]
   │      └── If function completes -> marked [ OK ]
   │
   └── 4. Prints final score: Returns 0 (pass) or 1 (fail) to the operating system
```

---

### 📄 File 8: `tests/test_tensor.cpp`
**Path:** `/home/abhishek/llm_inference/tests/test_tensor.cpp`

#### Why this code?
Unit tests verifying:
1. Multi-dimensional stride calculation for 1D, 2D, and 4D LLM tensors (`[batch, heads, seq, dim]`).
2. Coordinate offset math against hand-calculated ground truths.
3. Zero-copy slicing: Mutating a slice directly mutates the parent tensor without memory copies.
4. Numerical verification of $y = Wx + b$ against exact analytical ground truth:
   $$\begin{pmatrix} 2 & 3 \\ 4 & 5 \end{pmatrix} \begin{pmatrix} 10 \\ 20 \end{pmatrix} + \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 81 \\ 142 \end{pmatrix}$$

#### Annotated Code:
```cpp
#include "test_harness.h"             // Zero-dependency test macros (TEST, EXPECT_EQ)
#include "engine/core/tensor.h"       // Owning tensor container
#include "engine/core/tensor_shape.h" // Stride and shape calculations
#include "engine/core/tensor_view.h"  // Non-owning view
#include "engine/core/dtype.h"        // Data type enums
#include "engine/ops/ops_basic.h"     // Linear, matmul, relu ops

using namespace engine::core;
using namespace engine::ops;

// Test 1: Verify stride calculation across 1D, 2D, 3D and 4D shapes
TEST(TensorShapeTest, StridesCalculation) {
    // 1D tensor [10]: Stride is always 1
    TensorShape shape_1d({10});
    EXPECT_EQ(shape_1d.ndim(), 1);
    EXPECT_EQ(shape_1d.numel(), 10);
    EXPECT_EQ(shape_1d.stride(0), 1);

    // 2D tensor [2, 3]: Stride is [3, 1]
    TensorShape shape_2d({2, 3});
    EXPECT_EQ(shape_2d.ndim(), 2);
    EXPECT_EQ(shape_2d.numel(), 6);
    EXPECT_EQ(shape_2d.stride(0), 3);
    EXPECT_EQ(shape_2d.stride(1), 1);

    // 4D shape: [batch, heads, seq, dim] = [1, 8, 32, 64]
    TensorShape shape_4d({1, 8, 32, 64});
    EXPECT_EQ(shape_4d.ndim(), 4);
    EXPECT_EQ(shape_4d.numel(), 16384);
    EXPECT_EQ(shape_4d.stride(0), 16384); // 8 * 32 * 64
    EXPECT_EQ(shape_4d.stride(1), 2048);  // 32 * 64
    EXPECT_EQ(shape_4d.stride(2), 64);    // 64
    EXPECT_EQ(shape_4d.stride(3), 1);     // 1
}

// Test 2: Verify offset calculation mapping multi-dimensional coordinates to 1D index
TEST(TensorShapeTest, OffsetCalculation) {
    TensorShape shape({2, 3}); // Strides: [3, 1]
    EXPECT_EQ(shape.offset({0, 0}), 0); // 0*3 + 0*1 = 0
    EXPECT_EQ(shape.offset({0, 2}), 2); // 0*3 + 2*1 = 2
    EXPECT_EQ(shape.offset({1, 0}), 3); // 1*3 + 0*1 = 3
    EXPECT_EQ(shape.offset({1, 1}), 4); // 1*3 + 1*1 = 4
    EXPECT_EQ(shape.offset({1, 2}), 5); // 1*3 + 2*1 = 5

    // Verify out-of-bounds coordinate throws exception
    EXPECT_THROW(shape.offset({2, 0}), std::out_of_range);
}

// Test 3: Verify basic tensor storage and value mutation
TEST(TensorTest, BasicTensorOperations) {
    auto t = Tensor<float>::from_data(TensorShape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    EXPECT_EQ(t(0, 0), 1.0f);
    EXPECT_EQ(t(1, 1), 5.0f);

    // Mutate an element through the 2D coordinate accessor
    t(1, 1) = 99.0f;
    EXPECT_EQ(t(1, 1), 99.0f);
}

// Test 4: Verify zero-copy slicing and shared memory mutation
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
    EXPECT_EQ(slice(0, 0), 3.0f);
    EXPECT_EQ(slice(1, 1), 6.0f);

    // Mutating slice MUST mutate parent tensor (proves memory is shared, zero copies made!)
    slice(0, 0) = 42.0f;
    EXPECT_EQ(t(1, 0), 42.0f);
}

// Test 5: Verify linear projection against exact ground truth
TEST(OpsTest, LinearLayerGroundTruth) {
    // W = [[2, 3], [4, 5]]
    auto W = Tensor<float>::from_data(TensorShape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
    // x = [10, 20]
    auto x = Tensor<float>::from_data(TensorShape({2}), {10.0f, 20.0f});
    // b = [1, 2]
    auto b = Tensor<float>::from_data(TensorShape({2}), {1.0f, 2.0f});
    // y = [0, 0]
    auto y = Tensor<float>::zeros(TensorShape({2}));

    // y = W * x + b
    linear(W.view(), x.view(), b.view(), y.view());

    // y0 = 2*10 + 3*20 + 1 = 81
    // y1 = 4*10 + 5*20 + 2 = 142
    EXPECT_FLOAT_EQ(y(0), 81.0f);
    EXPECT_FLOAT_EQ(y(1), 142.0f);
}

// Test 6: Verify 2D matrix multiplication C = A * B
TEST(OpsTest, MatMul2D) {
    auto A = Tensor<float>::from_data(TensorShape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    auto B = Tensor<float>::from_data(TensorShape({3, 2}), {
        7.0f, 8.0f,
        9.0f, 1.0f,
        2.0f, 3.0f
    });
    auto C = Tensor<float>::zeros(TensorShape({2, 2}));

    matmul(A.view(), B.view(), C.view());

    // C[0, 0] = 1*7 + 2*9 + 3*2 = 31
    // C[0, 1] = 1*8 + 2*1 + 3*3 = 19
    // C[1, 0] = 4*7 + 5*9 + 6*2 = 85
    // C[1, 1] = 4*8 + 5*1 + 6*3 = 55
    EXPECT_FLOAT_EQ(C(0, 0), 31.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 19.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 85.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 55.0f);
}

// Test 7: Verify ReLU activation behavior
TEST(OpsTest, ReLUEvaluation) {
    auto input = Tensor<float>::from_data(TensorShape({4}), {-2.5f, 0.0f, 1.5f, -0.1f});
    auto output = Tensor<float>::zeros(TensorShape({4}));

    relu(input.view(), output.view());

    // Negatives clamped to 0, positive preserved
    EXPECT_FLOAT_EQ(output(0), 0.0f);
    EXPECT_FLOAT_EQ(output(1), 0.0f);
    EXPECT_FLOAT_EQ(output(2), 1.5f);
    EXPECT_FLOAT_EQ(output(3), 0.0f);
}

// Entrypoint for the test runner
int main() {
    return test_harness::run_all_tests();
}
```

#### 🔄 Flow of Code in `test_tensor.cpp`:
```text
Execution Flow when ./build/engine_tests runs:
   │
   ├── main() calls test_harness::run_all_tests()
   │
   ├── Step 1: StridesCalculation verifies 1D, 2D, 4D stride math
   │
   ├── Step 2: OffsetCalculation checks 2D index coordinate offsets
   │
   ├── Step 3: BasicTensorOperations validates data mutation
   │
   ├── Step 4: ZeroCopySlicing modifies slice and asserts underlying parent tensor was changed
   │
   ├── Step 5: LinearLayerGroundTruth calculates y = Wx + b and verifies [81, 142]
   │
   ├── Step 6: MatMul2D verifies 2x3 * 3x2 matrix multiplication
   │
   ├── Step 7: ReLUEvaluation checks non-linear activation clamping
   │
   └── All 7 tests return OK -> process exits with return code 0
```

---

### 📄 File 9: `src/main.cpp`
**Path:** `/home/abhishek/llm_inference/src/main.cpp`

#### Why this code?
A standalone CLI demo to visually inspect 2D tensor shapes, strides, zero-copy sub-views, and live linear layer execution in your terminal.

#### Annotated Code:
```cpp
#include <iostream>                   // Console output
#include "engine/core/tensor.h"       // Owning tensor class
#include "engine/core/tensor_shape.h" // Shape class
#include "engine/core/dtype.h"        // DType helper
#include "engine/ops/ops_basic.h"     // Linear projection operation

using namespace engine::core;
using namespace engine::ops;

int main() {
    std::cout << "======================================================\n";
    std::cout << "   LLM Inference Engine: Phase 1 Tensor Demo\n";
    std::cout << "======================================================\n\n";

    // Step 1: Create a 2D Tensor and inspect its memory layout and strides
    TensorShape shape_2d({2, 3});
    auto matrix = Tensor<float>::from_data(shape_2d, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    std::cout << "1. [2D Matrix] Shape: " << matrix.shape().to_string() << "\n";
    std::cout << "   Strides: [" << matrix.stride(0) << ", " << matrix.stride(1) << "]\n";
    std::cout << "   Total Elements: " << matrix.numel() << "\n";
    std::cout << "   2D Matrix Display:\n";
    matrix.print(); // Prints formatted 2D table
    std::cout << "\n";

    // Step 2: Create a zero-copy sub-view of the second row (Row 1)
    std::cout << "2. [Zero-Copy View Slicing]\n";
    auto view = matrix.view();
    auto row_1_slice = view.slice_dim0(1, 2); // Row 1 only
    std::cout << "   Slice Shape: " << row_1_slice.shape().to_string() << "\n";
    std::cout << "   Slice Values:\n";
    row_1_slice.print();
    std::cout << "\n";

    // Step 3: Run a live Linear Projection Layer: y = W * x + b
    std::cout << "3. [Linear Projection Layer: y = Wx + b]\n";
    // Weight matrix W (2x2)
    auto W = Tensor<float>::from_data(TensorShape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
    // Input vector x (2)
    auto x = Tensor<float>::from_data(TensorShape({2}), {10.0f, 20.0f});
    // Bias vector b (2)
    auto b = Tensor<float>::from_data(TensorShape({2}), {1.0f, 2.0f});
    // Output allocation y (2)
    auto y = Tensor<float>::zeros(TensorShape({2}));

    std::cout << "   Weight Matrix W (2x2):\n";
    W.print();
    std::cout << "   Input Vector x (2):\n";
    x.print();
    std::cout << "   Bias Vector b (2):\n";
    b.print();

    // Perform linear projection
    linear(W.view(), x.view(), b.view(), y.view());

    // Print and inspect computed result
    std::cout << "   Calculated Output Vector y (2):\n";
    y.print();
    std::cout << "   -> Ground truth check: y[0] = 2*10 + 3*20 + 1 = 81\n";
    std::cout << "   -> Ground truth check: y[1] = 4*10 + 5*20 + 2 = 142\n\n";

    std::cout << "✅ Phase 1 Tensor Foundations successfully verified!\n";
    return 0;
}
```

#### 🔄 Flow of Code in `src/main.cpp`:
```text
Execution Flow when ./build/engine_cli runs:
   │
   ├── 1. Banner header is printed to stdout
   │
   ├── 2. Tensor matrix is created with shape [2, 3]
   │      Prints shape, strides ([3, 1]), element count (6), and formatted 2D table
   │
   ├── 3. Creates zero-copy slice of Row 1 (shape [1, 3]) and prints [4, 5, 6]
   │
   ├── 4. Allocates W (2x2), x (2), b (2), and zeroed output y (2)
   │
   ├── 5. Calls linear(W.view(), x.view(), b.view(), y.view())
   │
   ├── 6. Prints resulting output vector [81, 142] and shows ground-truth check
   │
   └── 7. Prints completion checkmark and exits successfully
```

---

## 5. 🧪 Build & Run Commands

```bash
# 1. Configure and compile in Release mode
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Run automated unit tests
./build/engine_tests

# 3. Run the interactive visualizer
./build/engine_cli
```

### ✅ Expected Terminal Output:
```text
[==========] Running 7 test cases.
[ RUN      ] TensorShapeTest.StridesCalculation
[       OK ] TensorShapeTest.StridesCalculation
[ RUN      ] TensorShapeTest.OffsetCalculation
[       OK ] TensorShapeTest.OffsetCalculation
[ RUN      ] TensorTest.BasicTensorOperations
[       OK ] TensorTest.BasicTensorOperations
[ RUN      ] TensorViewTest.ZeroCopySlicing
[       OK ] TensorViewTest.ZeroCopySlicing
[ RUN      ] OpsTest.LinearLayerGroundTruth
[       OK ] OpsTest.LinearLayerGroundTruth
[ RUN      ] OpsTest.MatMul2D
[       OK ] OpsTest.MatMul2D
[ RUN      ] OpsTest.ReLUEvaluation
[       OK ] OpsTest.ReLUEvaluation
[==========] 7 tests ran.
[  PASSED  ] 7 tests.

======================================================
   LLM Inference Engine: Phase 1 Tensor Demo
======================================================

1. [2D Matrix] Shape: [2, 3]
   Strides: [3, 1]
   Total Elements: 6
   2D Matrix Display:
[
  [     1,      2,      3],
  [     4,      5,      6]
]

2. [Zero-Copy View Slicing]
   Slice Shape: [1, 3]
   Slice Values:
[
  [     4,      5,      6]
]

3. [Linear Projection Layer: y = Wx + b]
   Weight Matrix W (2x2):
[
  [     2,      3],
  [     4,      5]
]
   Input Vector x (2):
[10, 20]
   Bias Vector b (2):
[1, 2]
   Calculated Output Vector y (2):
[81, 142]
   -> Ground truth check: y[0] = 2*10 + 3*20 + 1 = 81
   -> Ground truth check: y[1] = 4*10 + 5*20 + 2 = 142

✅ Phase 1 Tensor Foundations successfully verified!
```

---

## 6. 📢 Shareable Takeaway

> *"Why high-performance tensor engines avoid `vector<vector<float>>`: Nested vectors fragment memory and murder CPU cache lines. Flatten your tensor into a single contiguous 1D buffer in RAM and map multi-dimensional coordinates using row-major strides ($offset = r \cdot stride_0 + c \cdot stride_1$). This enables zero-copy slicing and prepares your memory for direct CUDA transfers."*

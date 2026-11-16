# Phase 1: Quick Reference & Cheat Sheet
*Direct, zero-fluff summary of Phase 1 concepts, terms, and formulas.*

---

## 1. Core Problem: Flat RAM vs. Multi-Dim Tensors

- **RAM is 1D:** Hardware memory is a flat line of bytes.
- **Why NOT `std::vector<std::vector<float>>`:**
  1. **Cache thrashing:** Every row is scattered at random heap addresses. CPUs fetch contiguous 64-byte lines; pointer chasing kills speed.
  2. **Zero-copy impossible:** Slicing a sub-matrix forces deep copies.
  3. **No GPU support:** `cudaMemcpy` requires a single contiguous `float*` pointer.
- **The Solution:** Store everything in **one flat 1D array** (`std::vector<T>`) and use math (**strides**) to access `(row, col)`.

---

## 2. Key Terms & Formulas

| Term | What It Is | Formula / Example |
|---|---|---|
| **Shape (`dims`)** | Tensor dimensions | e.g. `[2, 3]` (2 rows, 3 columns) |
| **Rank (`ndim`)** | Number of axes / dimensions | `dims.size()` (for `[2, 3]`, rank is 2) |
| **Numel (`numel`)** | Total number of elements | $\prod \text{dims}_i = 2 \times 3 = 6$ |
| **Stride** | Steps in 1D RAM to advance by 1 along a dimension | Row-major: step right-to-left |
| **Offset** | 1D index in flat RAM for coordinates $(i_0, i_1, \dots)$ | $\text{offset} = \sum i_d \times \text{stride}_d$ |

### Stride Calculation (Row-Major Contiguous)
- Last dimension: $\text{stride}_{D-1} = 1$
- Prior dimensions: $\text{stride}_k = \text{stride}_{k+1} \times \text{dim}_{k+1}$

**Example for `shape = [2, 3]`:**
- `stride[1] = 1`
- `stride[0] = stride[1] * dim[1] = 1 * 3 = 3`
- $\rightarrow \text{strides} = [3, 1]$

**Accessing coordinate `(r=1, c=1)`:**
$$\text{offset}(1, 1) = 1 \times 3 + 1 \times 1 = \mathbf{4}$$

---

## 3. Architecture: Owning vs. Non-Owning

```text
┌─────────────────────────────────┐
│        Tensor<T> (Owning)       │
│  Owns heap std::vector<T>       │
│  Manages memory lifetime (RAII) │
└──────────────┬──────────────────┘
               │ .view() or implicit conversion
               ▼
┌─────────────────────────────────┐
│      TensorView<T> (Non-owning) │
│  Holds raw pointer: T* data_    │
│  Holds TensorShape: shape_      │
│  Zero heap allocations          │
│  Fast slicing & operator inputs │
└─────────────────────────────────┘
```

- **`Tensor<T>`:** Allocates memory. When it leaves scope, memory frees automatically.
- **`TensorView<T>`:** Zero-cost lens over existing memory.
  - Slicing a row: `new_data = data_ + (row_idx * stride[0])`. No data copied!

---

## 4. Fundamental Operator: Linear Projection

$$y = Wx + b$$

- $W \in \mathbb{R}^{M \times K}$ (Weights)
- $x \in \mathbb{R}^{K}$ (Input Vector)
- $b \in \mathbb{R}^{M}$ (Bias)
- $y \in \mathbb{R}^{M}$ (Output Vector)

### Per-Element Formula:
$$y_i = \sum_{j=0}^{K-1} W_{i, j} \cdot x_j + b_i$$

**Ground Truth Example:**
$$W = \begin{pmatrix} 2 & 3 \\ 4 & 5 \end{pmatrix}, \quad x = \begin{pmatrix} 10 \\ 20 \end{pmatrix}, \quad b = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

$$y_0 = 2(10) + 3(20) + 1 = 81$$
$$y_1 = 4(10) + 5(20) + 2 = 142$$

---

## 5. File Responsibilities in 1 Line Each

- **`CMakeLists.txt`**: Flags (`-O3`, `-mavx2`, `-mfma`), C++20 standard, build targets.
- **`dtype.h`**: `DType` enum (`Float32`, `Int8`, etc.) and `dtype_size()` byte lookups.
- **`tensor_shape.h`**: Computes strides right-to-left and translates `(r, c)` $\rightarrow$ flat 1D offset.
- **`tensor_view.h`**: Non-owning `(T*, TensorShape)`. Enables zero-copy `slice_dim0()`.
- **`tensor.h`**: Owning buffer (`std::vector<T>`) + view conversions.
- **`ops_basic.h`**: Pure functions for $y = Wx + b$, MatMul ($C = AB$), ReLU, Add.
- **`test_harness.h`**: Zero-dependency test runner (`TEST`, `EXPECT_EQ`).
- **`test_tensor.cpp`**: 7 unit tests verifying stride math, slicing, and $y = Wx + b$.
- **`main.cpp`**: CLI demo showing formatted prints and live calculation.

---

## 6. Commands Cheat Sheet

```bash
# Build (Release mode with optimizations)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
./build/engine_tests

# Run interactive demo
./build/engine_cli
```

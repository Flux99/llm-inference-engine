# llm-inference-engine

> A high-performance Large Language Model inference engine built from scratch in modern **C++20** (with companion **TypeScript** implementations for low-level systems mental modeling).

---

## 🎯 Vision & Goals

Modern LLM inference is governed by systems constraints: memory bandwidth, CPU cache locality, SIMD parallelism, kernel launch overheads, and GPU memory hierarchies.

This project implements an end-to-end inference runtime from first principles:
- **Track 1:** C++ Foundations & Graph Engine (Tensor memory layouts, Model Loaders, DAG TopoSort)
- **Track 2:** Systems Performance & CPU Optimization (Cache Tiling, AVX2/FMA SIMD, OpenMP)

---

## 🗺️ Project Status & Roadmap

```text
[x] Phase 1: Tensor Foundations, Strides & Memory Layout (C++20 & TypeScript)
[ ] Phase 2: Model Formats & Weight Loaders (SafeTensors mmap & ONNX Protobuf) ◄ (Up Next / Thinking Phase)
```

---

## ✅ Current Milestone: Phase 1 (Completed)

Implemented core multidimensional tensor abstractions without relying on nested vectors (`std::vector<std::vector<float>>`):

- **Contiguous 1D Memory:** Stores all elements in a single contiguous buffer for sequential cache line fills (64-byte L1 cache alignment).
- **Row-Major Strides:** Fast index translation from coordinates to memory offsets:
  $$\text{offset}(r, c) = r \times \text{stride}_0 + c \times \text{stride}_1$$
- **Zero-Copy Views (`TensorView`):** Slicing sub-tensors (e.g. attention heads, KV caches) without allocating new heap memory.
- **Dual C++ & TypeScript Rosetta Stone:** Every low-level C++ abstraction is mirrored in TypeScript using `ArrayBuffer` and `Float32Array` views.

---

## 🔮 What I'm Building Next: Phase 2 (Model Weight Loaders)

> **Current Thinking & Architectural Plan:**
> 
> Billions of model weights cannot be loaded with Python `pickle` (security flaws & 2x memory duplication) or naive `read()` syscalls (which stall the CPU while copying gigabytes into heap RAM).
>
> In Phase 2, I am planning to implement zero-overhead loaders for the two industry-standard formats:
> 1. **SafeTensors Zero-Copy Loading (`mmap`):**
>    - Leverage POSIX `mmap()` / Windows `MapViewOfFile` to map disk pages straight into the process virtual address space in $< 1$ ms.
>    - Parse the 8-byte uint64 header size and UTF-8 JSON tensor metadata.
>    - Point non-owning `TensorView<float>` objects directly into mapped memory pages—letting the OS kernel page in weights on demand with zero byte copies.
> 2. **ONNX Protobuf Graph & Weight Extraction:**
>    - Use Google Protocol Buffers (`protoc`) to parse `.onnx` computational graphs.
>    - Inspect `NodeProto` operator topologies (Gemm, Relu, Add) and extract `TensorProto` initializers directly into `Tensor<float>` buffers.

---

## 🛠️ Project Structure

```text
llm-inference-engine/
├── include/engine/
│   ├── core/              # Core abstractions
│   │   ├── dtype.h        # Strongly-typed precisions (FP32, FP16, INT8, INT32)
│   │   ├── tensor_shape.h # Dimensions, strides & offset calculation
│   │   ├── tensor_view.h  # Non-owning zero-copy window
│   │   └── tensor.h       # RAII owning contiguous tensor
│   └── ops/
│       └── ops_basic.h    # Baseline math operators (MatMul2D, Linear, ReLU)
├── src/
│   └── main.cpp           # Interactive CLI visualization demo
├── tests/
│   ├── test_harness.h     # Minimal dependency-free test framework
│   └── test_tensor.cpp    # Unit & numerical ground-truth tests
├── ts/                    # TypeScript mirror implementation
│   ├── src/               # TS source files (tensor, shape, views, ops)
│   └── tests/             # node:test test suite
├── docs/course/           # Detailed phase-by-phase implementation guides
└── CMakeLists.txt         # Modern C++20 build configuration (-O3, -mavx2, -mfma)
```

---

## 🚀 Quickstart: Build & Run

### Prerequisites
- C++20 compliant compiler (`g++` 11+ or `clang` 14+)
- CMake 3.20+
- Node.js 18+ (for TypeScript mirror)

### 1. Build & Run C++ Engine
```bash
# Configure CMake with -O3 and AVX2 optimizations
cmake -B build

# Compile tests and CLI executable
cmake --build build -j$(nproc)

# Run automated C++ unit tests
./build/engine_tests

# Run interactive CLI demo
./build/engine_cli
```

### 2. Run TypeScript Mirror
```bash
# Run unit tests
npm test

# Run TypeScript interactive demo
npm start
```

---

## 📄 License
MIT

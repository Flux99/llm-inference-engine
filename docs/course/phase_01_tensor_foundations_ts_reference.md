# Phase 1: TypeScript Translation & Mental Model Rosetta Stone
*Bridging C++20 Low-Level Systems to Modern TypeScript*

As a TypeScript developer, you already understand objects, arrays, typing, and functions. In C++, we do the **exact same logical tasks**, but instead of letting the V8 JavaScript engine handle memory and garbage collection behind the scenes, **we control the raw memory bytes directly**.

This reference document translates every single C++ file in Phase 1 into idiomatic TypeScript, side-by-side, so you can mentally map:
> *"Oh! In TypeScript I write `X`, which means in C++ I am writing `Y` to control memory directly."*

---

## 🗺️ High-Level Concept Translation Table

| C++ Concept in Phase 1 | TypeScript Equivalent | What's Really Happening? |
|---|---|---|
| `std::vector<float>` | `Float32Array` (backed by `ArrayBuffer`) | A contiguous chunk of 32-bit floats in memory without object wrappers. |
| `float* data_ptr` | Byte offset / index into `ArrayBuffer` | Direct memory address pointing to the first number in RAM. |
| `TensorShape` | Class / Object `{ dims: number[], strides: number[] }` | Metadata describing dimensions and step sizes. |
| `TensorView<T>` | `new Float32Array(buffer, byteOffset, length)` | A zero-copy view/window over an existing buffer! |
| `Tensor<T>` | Class owning an `ArrayBuffer` & `Float32Array` | The owner of the memory lifecycle (RAII in C++, GC in TS). |
| `operator()(r, c)` | Method `at(r, c)` or `get(r, c)` | Index math: `r * stride[0] + c * stride[1]`. |
| `CMakeLists.txt` | `package.json` + `tsconfig.json` | Project setup, compiler flags, and target definitions. |
| `test_harness.h` | `jest` / `vitest` / `node:test` (`describe`, `it`, `expect`) | Unit test assertion runner. |

---

## 📁 File-by-File Comparison & Translation

---

### 📄 File 1: Build & Config (`CMakeLists.txt` vs. `tsconfig.json` / `package.json`)

In TypeScript/Node.js, you configure compiler options with `tsconfig.json` and build scripts with `package.json`. In C++, CMake is both: it declares the compiler flags, dependencies, and build targets.

```json
// package.json / tsconfig.json equivalent of CMakeLists.txt
{
  "name": "llm-inference-engine",
  "version": "1.0.0",
  "compilerOptions": {
    "target": "ES2022",              // Like CMAKE_CXX_STANDARD 20
    "module": "NodeNext",
    "strict": true,                  // Like -Wall -Wextra -Wpedantic
    "rootDir": "src",
    "outDir": "dist"
  },
  "scripts": {
    "build": "tsc",                  // Like cmake --build build
    "test": "node --test dist/tests" // Like ctest
  }
}
```

---

### 📄 File 2: Data Types (`dtype.h` vs. `dtype.ts`)

#### 🔷 TypeScript Equivalent (`dtype.ts`):
```typescript
/**
 * In TypeScript, all regular numbers are 64-bit IEEE-754 floats.
 * But when dealing with TypedArrays (WebAssembly / WebGPU / Buffers),
 * we distinguish Float32, Float16, Int8, and Int32.
 */

export enum DType {
  Float32 = "Float32",
  Float16 = "Float16",
  Int8 = "Int8",
  Int32 = "Int32",
}

// Equivalent to C++: inline size_t dtype_size(DType dtype)
export function dtypeSize(dtype: DType): number {
  switch (dtype) {
    case DType.Float32: return 4; // 4 bytes (32 bits)
    case DType.Float16: return 2; // 2 bytes (16 bits)
    case DType.Int8:    return 1; // 1 byte  (8 bits)
    case DType.Int32:   return 4; // 4 bytes (32 bits)
    default:
      throw new Error(`Unknown DType: ${dtype}`);
  }
}

// In TS, we don't need C++ template specializations like TypeToDType<T>
// because TS uses union types or TypedArray constructor types:
export type DTypeTypedArrayMap = {
  [DType.Float32]: Float32Array;
  [DType.Int8]: Int8Array;
  [DType.Int32]: Int32Array;
};
```

#### 💡 The Takeaway:
In C++, `dtype_size` tells us how many raw bytes to allocate in RAM (`malloc(numel * 4)`). In TypeScript, `Float32Array.BYTES_PER_ELEMENT` is the exact same thing (4).

---

### 📄 File 3: Tensor Shape & Strides (`tensor_shape.h` vs. `tensor_shape.ts`)

This is the exact same math in both languages!

#### 🔷 TypeScript Equivalent (`tensor_shape.ts`):
```typescript
export class TensorShape {
  readonly dims: number[];
  readonly strides: number[];
  readonly numel: number;

  constructor(dims: number[]) {
    this.dims = [...dims];

    if (this.dims.length === 0) {
      this.strides = [1];
      this.numel = 0;
      return;
    }

    // 1. Calculate total number of elements (multiplication of all dimensions)
    this.numel = this.dims.reduce((acc, d) => acc * d, 1);

    // 2. Compute Row-Major Contiguous Strides from right to left
    // Formula:
    // strides[last] = 1
    // strides[k] = strides[k+1] * dims[k+1]
    this.strides = new Array(this.dims.length).fill(0);
    this.strides[this.strides.length - 1] = 1;

    for (let k = this.dims.length - 2; k >= 0; k--) {
      this.strides[k] = this.strides[k + 1] * this.dims[k + 1];
    }
  }

  get ndim(): number {
    return this.dims.length;
  }

  dim(i: number): number {
    if (i < 0 || i >= this.dims.length) throw new RangeError("Dimension index out of bounds");
    return this.dims[i];
  }

  stride(i: number): number {
    if (i < 0 || i >= this.strides.length) throw new RangeError("Stride index out of bounds");
    return this.strides[i];
  }

  // Maps coordinates [row, col] to 1D index
  offset(...indices: number[]): number {
    if (indices.length !== this.dims.length) {
      throw new Error(`Expected ${this.dims.length} indices, got ${indices.length}`);
    }

    let off = 0;
    for (let d = 0; d < indices.length; d++) {
      const idx = indices[d];
      if (idx < 0 || idx >= this.dims[d]) {
        throw new RangeError(`Index ${idx} out of bounds for dim ${d} (size ${this.dims[d]})`);
      }
      off += idx * this.strides[d];
    }
    return off;
  }

  equals(other: TensorShape): boolean {
    if (this.dims.length !== other.dims.length) return false;
    return this.dims.every((d, i) => d === other.dims[i]);
  }

  toString(): string {
    return `[${this.dims.join(", ")}]`;
  }
}
```

#### 💡 The Takeaway:
Notice how the logic is **identical**. In C++, `offset()` uses pointer arithmetic to access the memory. In TypeScript, `offset()` calculates the integer index into a `Float32Array`.

---

### 📄 File 4: Non-Owning View (`tensor_view.h` vs. `tensor_view.ts`)

In C++, `TensorView` points to a raw `float*` and a `TensorShape`.
In TypeScript, **`TypedArray.subarray()`** is the exact same concept! It does **not** copy memory; it creates a new window pointing to the same underlying `ArrayBuffer`.

#### 🔷 TypeScript Equivalent (`tensor_view.ts`):
```typescript
import { TensorShape } from "./tensor_shape";

export class TensorView {
  // data is a TypedArray viewing an underlying ArrayBuffer
  // byteOffset gives the start in memory without duplicating bytes!
  readonly data: Float32Array;
  readonly shape: TensorShape;

  constructor(data: Float32Array, shape: TensorShape) {
    this.data = data;
    this.shape = shape;
  }

  get numel(): number { return this.shape.numel; }
  get ndim(): number { return this.shape.ndim; }
  dim(i: number): number { return this.shape.dim(i); }
  stride(i: number): number { return this.shape.stride(i); }

  // Fast 1D access: t.at1D(i)
  at1D(i: number): number {
    return this.data[i * this.shape.stride(0)];
  }

  set1D(i: number, val: number): void {
    this.data[i * this.shape.stride(0)] = val;
  }

  // Fast 2D access: t.at2D(row, col)
  // In C++ this is: operator()(size_t r, size_t c)
  at2D(r: number, c: number): number {
    return this.data[r * this.shape.stride(0) + c * this.shape.stride(1)];
  }

  set2D(r: number, c: number, val: number): void {
    this.data[r * this.shape.stride(0) + c * this.shape.stride(1)] = val;
  }

  // General multi-dim access
  at(...indices: number[]): number {
    return this.data[this.shape.offset(...indices)];
  }

  // Zero-Copy Slice: Just like C++ slice_dim0!
  // In JS/TS: Float32Array.subarray(begin, end) creates a view over the SAME buffer!
  sliceDim0(start: number, end: number): TensorView {
    if (start >= end || end > this.shape.dim(0)) {
      throw new RangeError("Invalid slice bounds");
    }

    const newDims = [...this.shape.dims];
    newDims[0] = end - start;

    const startOffset = start * this.shape.stride(0);
    const endOffset = startOffset + (end - start) * this.shape.stride(0);

    // ZERO MEMORY COPIES: Creates a sub-view over the exact same ArrayBuffer!
    const subArray = this.data.subarray(startOffset, endOffset);
    return new TensorView(subArray, new TensorShape(newDims));
  }

  print(): void {
    if (this.ndim === 1) {
      console.log(`[${Array.from(this.data).join(", ")}]`);
    } else if (this.ndim === 2) {
      const rows: string[] = [];
      for (let r = 0; r < this.dim(0); r++) {
        const rowVals: string[] = [];
        for (let c = 0; c < this.dim(1); c++) {
          rowVals.push(this.at2D(r, c).toFixed(4).padStart(8));
        }
        rows.push(`  [${rowVals.join(", ")}]`);
      }
      console.log(`[\n${rows.join(",\n")}\n]`);
    } else {
      console.log(`TensorView(shape=${this.shape.toString()})`);
    }
  }
}
```

#### 💡 The Takeaway:
* In C++: `T* new_data = data_ + (start * stride(0));`
* In TS: `const subArray = this.data.subarray(startOffset, endOffset);`
Both do the exact same thing: **zero-copy slicing**.

---

### 📄 File 5: Owning Tensor (`tensor.h` vs. `tensor.ts`)

In C++, `Tensor<T>` owns the heap allocation via `std::vector<T>`.
In TypeScript, `Tensor` owns the newly allocated `Float32Array`.

#### 🔷 TypeScript Equivalent (`tensor.ts`):
```typescript
import { TensorShape } from "./tensor_shape";
import { TensorView } from "./tensor_view";

export class Tensor {
  readonly shape: TensorShape;
  // This owns the memory buffer!
  private readonly data_: Float32Array;

  constructor(shape: TensorShape, fillValue: number = 0) {
    this.shape = shape;
    // Allocates a single continuous block of 32-bit floats in RAM:
    this.data_ = new Float32Array(this.shape.numel);
    if (fillValue !== 0) {
      this.data_.fill(fillValue);
    }
  }

  // Factory: Tensor.zeros([2, 3])
  static zeros(shape: TensorShape): Tensor {
    return new Tensor(shape, 0);
  }

  // Factory: Tensor.ones([2, 3])
  static ones(shape: TensorShape): Tensor {
    return new Tensor(shape, 1);
  }

  // Factory: Tensor.fromData([2, 3], [1, 2, 3, 4, 5, 6])
  static fromData(shape: TensorShape, data: number[]): Tensor {
    if (data.length !== shape.numel) {
      throw new Error(`Data size (${data.length}) doesn't match shape numel (${shape.numel})`);
    }
    const t = new Tensor(shape);
    t.data_.set(data);
    return t;
  }

  get data(): Float32Array { return this.data_; }
  get numel(): number { return this.shape.numel; }
  get ndim(): number { return this.shape.ndim; }
  dim(i: number): number { return this.shape.dim(i); }
  stride(i: number): number { return this.shape.stride(i); }

  at2D(r: number, c: number): number {
    return this.data_[r * this.shape.stride(0) + c * this.shape.stride(1)];
  }

  set2D(r: number, c: number, val: number): void {
    this.data_[r * this.shape.stride(0) + c * this.shape.stride(1)] = val;
  }

  // Creates a lightweight non-owning view over this tensor's memory
  view(): TensorView {
    return new TensorView(this.data_, this.shape);
  }

  print(): void {
    this.view().print();
  }
}
```

---

### 📄 File 6: Basic Math & Linear Layer (`ops_basic.h` vs. `ops_basic.ts`)

Here are the mathematical kernels: Elementwise Add, ReLU, 2D Matrix Multiplication ($C = AB$), and the Linear Layer ($y = Wx + b$).

#### 🔷 TypeScript Equivalent (`ops_basic.ts`):
```typescript
import { TensorView } from "./tensor_view";

// Elementwise Add: out = a + b
export function add(a: TensorView, b: TensorView, out: TensorView): void {
  if (!a.shape.equals(b.shape) || !a.shape.equals(out.shape)) {
    throw new Error("Shape mismatch in add operation");
  }
  for (let i = 0; i < a.numel; i++) {
    out.data[i] = a.data[i] + b.data[i];
  }
}

// Elementwise ReLU: out = max(0, in)
export function relu(input: TensorView, out: TensorView): void {
  if (!input.shape.equals(out.shape)) {
    throw new Error("Shape mismatch in relu operation");
  }
  for (let i = 0; i < input.numel; i++) {
    out.data[i] = Math.max(0, input.data[i]);
  }
}

// 2D Matrix Multiplication: C(M, N) = A(M, K) * B(K, N)
export function matmul(A: TensorView, B: TensorView, C: TensorView): void {
  if (A.ndim !== 2 || B.ndim !== 2 || C.ndim !== 2) {
    throw new Error("matmul expects 2D tensors");
  }

  const M = A.dim(0);
  const K = A.dim(1);
  const K2 = B.dim(0);
  const N = B.dim(1);

  if (K !== K2 || C.dim(0) !== M || C.dim(1) !== N) {
    throw new Error("Matrix dimension mismatch in matmul");
  }

  // Reset output to zero before accumulating
  C.data.fill(0);

  // Cache-friendly i-k-j loop ordering
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const a_ik = A.at2D(i, k);
      for (let j = 0; j < N; j++) {
        // C(i, j) += A(i, k) * B(k, j)
        const current = C.at2D(i, j);
        C.set2D(i, j, current + a_ik * B.at2D(k, j));
      }
    }
  }
}

// Linear Layer / Projection: y = W * x + b
// W: (M, K), x: (K), b: (M), y: (M)
export function linear(
  W: TensorView,
  x: TensorView,
  b: TensorView,
  y: TensorView
): void {
  if (W.ndim !== 2 || x.ndim !== 1 || b.ndim !== 1 || y.ndim !== 1) {
    throw new Error("Dimension mismatch in linear layer");
  }

  const M = W.dim(0); // Output features
  const K = W.dim(1); // Input features

  if (x.dim(0) !== K || b.dim(0) !== M || y.dim(0) !== M) {
    throw new Error("Shape mismatch in linear parameters");
  }

  for (let i = 0; i < M; i++) {
    let sum = b.at1D(i); // Start with bias b[i]
    for (let j = 0; j < K; j++) {
      sum += W.at2D(i, j) * x.at1D(j); // Dot product of row W[i, :] and vector x[:]
    }
    y.set1D(i, sum); // Write to output vector y[i]
  }
}
```

---

### 📄 File 7: Unit Tests (`test_tensor.cpp` vs. `test_tensor.test.ts`)

In C++, we wrote a tiny macro-based test runner (`TEST`, `EXPECT_EQ`).
In TypeScript, this is standard `node:test` or `vitest`/`jest` syntax (`describe`, `it`, `expect`).

#### 🔷 TypeScript Equivalent (`test_tensor.test.ts`):
```typescript
import { describe, it } from "node:test";
import assert from "node:assert";
import { TensorShape } from "./tensor_shape";
import { Tensor } from "./tensor";
import { linear, matmul, relu } from "./ops_basic";

describe("TensorShape Tests", () => {
  it("computes row-major strides correctly", () => {
    // 2D: [2, 3] -> strides: [3, 1]
    const shape2D = new TensorShape([2, 3]);
    assert.strictEqual(shape2D.stride(0), 3);
    assert.strictEqual(shape2D.stride(1), 1);

    // 4D (LLM tensor): [1, 8, 32, 64]
    const shape4D = new TensorShape([1, 8, 32, 64]);
    assert.strictEqual(shape4D.stride(0), 16384);
    assert.strictEqual(shape4D.stride(1), 2048);
    assert.strictEqual(shape4D.stride(2), 64);
    assert.strictEqual(shape4D.stride(3), 1);
  });

  it("calculates flat memory offsets", () => {
    const shape = new TensorShape([2, 3]);
    assert.strictEqual(shape.offset(0, 0), 0);
    assert.strictEqual(shape.offset(0, 2), 2);
    assert.strictEqual(shape.offset(1, 0), 3);
    assert.strictEqual(shape.offset(1, 1), 4);
    assert.strictEqual(shape.offset(1, 2), 5);
  });
});

describe("Zero-Copy Slicing Tests", () => {
  it("proves that mutating a slice mutates the original parent tensor", () => {
    const t = Tensor.fromData(new TensorShape([4, 2]), [
      1, 2,
      3, 4,
      5, 6,
      7, 8
    ]);

    const view = t.view();
    const slice = view.sliceDim0(1, 3); // Rows 1 to 2

    assert.strictEqual(slice.at2D(0, 0), 3);
    assert.strictEqual(slice.at2D(1, 1), 6);

    // Mutate through slice!
    slice.set2D(0, 0, 42);

    // Parent tensor at row 1, col 0 MUST be 42! (Zero-copy proof!)
    assert.strictEqual(t.at2D(1, 0), 42);
  });
});

describe("Ops Verification Tests", () => {
  it("computes linear layer: y = Wx + b exactly", () => {
    const W = Tensor.fromData(new TensorShape([2, 2]), [2, 3, 4, 5]);
    const x = Tensor.fromData(new TensorShape([2]), [10, 20]);
    const b = Tensor.fromData(new TensorShape([2]), [1, 2]);
    const y = Tensor.zeros(new TensorShape([2]));

    linear(W.view(), x.view(), b.view(), y.view());

    // y0 = 2*10 + 3*20 + 1 = 81
    // y1 = 4*10 + 5*20 + 2 = 142
    assert.strictEqual(y.data[0], 81);
    assert.strictEqual(y.data[1], 142);
  });
});
```

---

### 📄 File 8: Interactive CLI (`src/main.cpp` vs. `src/main.ts`)

#### 🔷 TypeScript Equivalent (`main.ts`):
```typescript
import { TensorShape } from "./tensor_shape";
import { Tensor } from "./tensor";
import { linear } from "./ops_basic";

function main() {
  console.log("======================================================");
  console.log("   LLM Inference Engine: Phase 1 TypeScript Demo");
  console.log("======================================================\n");

  // 1. Inspect 2D Tensor
  const shape2D = new TensorShape([2, 3]);
  const matrix = Tensor.fromData(shape2D, [1, 2, 3, 4, 5, 6]);

  console.log(`1. [2D Matrix] Shape: ${matrix.shape.toString()}`);
  console.log(`   Strides: [${matrix.stride(0)}, ${matrix.stride(1)}]`);
  console.log(`   Total Elements: ${matrix.numel}`);
  console.log("   2D Matrix Display:");
  matrix.print();
  console.log();

  // 2. Zero-Copy Slicing
  console.log("2. [Zero-Copy View Slicing]");
  const view = matrix.view();
  const row1Slice = view.sliceDim0(1, 2);
  console.log(`   Slice Shape: ${row1Slice.shape.toString()}`);
  console.log("   Slice Values:");
  row1Slice.print();
  console.log();

  // 3. Linear Projection: y = Wx + b
  console.log("3. [Linear Projection Layer: y = Wx + b]");
  const W = Tensor.fromData(new TensorShape([2, 2]), [2, 3, 4, 5]);
  const x = Tensor.fromData(new TensorShape([2]), [10, 20]);
  const b = Tensor.fromData(new TensorShape([2]), [1, 2]);
  const y = Tensor.zeros(new TensorShape([2]));

  console.log("   Weight Matrix W (2x2):");
  W.print();
  console.log("   Input Vector x (2):");
  x.print();
  console.log("   Bias Vector b (2):");
  b.print();

  linear(W.view(), x.view(), b.view(), y.view());

  console.log("   Calculated Output Vector y (2):");
  y.print();
  console.log("   -> Ground truth check: y[0] = 2*10 + 3*20 + 1 = 81");
  console.log("   -> Ground truth check: y[1] = 4*10 + 5*20 + 2 = 142\n");

  console.log("✅ Phase 1 Tensor Foundations successfully verified in TypeScript!");
}

main();
```

---

## 🎯 Summary: What Changes When We Move from TS to C++?

1. **Memory Allocation:**
   - In TS: `new Float32Array(n)` allocates memory managed by the V8 garbage collector.
   - In C++: `std::vector<float>` allocates memory directly on the OS heap via C++ allocator (`new`/`delete`), cleaned up automatically when the `Tensor` goes out of scope (RAII).

2. **Pointer Arithmetic:**
   - In TS: `array.subarray(start, end)` is an API call that returns a wrapper object with an internal byteOffset.
   - In C++: `T* new_ptr = data_ + offset;` is **one raw CPU instruction** adding an integer offset to a 64-bit hardware memory address.

3. **Performance:**
   - In TS: JIT compiler does a good job, but bounds checks, dynamic dispatch, and lack of direct SIMD/CUDA instructions limit matrix multiplication performance.
   - In C++: The compiler generates **AVX2 vector instructions** (`_mm256_fmadd_ps`) processing 8 numbers per CPU clock cycle, and later compiles directly into **NVIDIA CUDA machine code** on your GTX 1060.


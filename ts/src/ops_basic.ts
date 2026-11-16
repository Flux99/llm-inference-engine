import { TensorView } from "./tensor_view.ts";

/**
 * Basic Tensor Operators (Add, ReLU, MatMul, Linear Projection)
 */

export function add(a: TensorView, b: TensorView, out: TensorView): void {
  if (!a.shape.equals(b.shape) || !a.shape.equals(out.shape)) {
    throw new Error("Shape mismatch in add operation");
  }
  for (let i = 0; i < a.numel; i++) {
    out.data[i] = a.data[i] + b.data[i];
  }
}

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

  C.data.fill(0);

  // Cache-friendly i-k-j loop ordering
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const a_ik = A.at2D(i, k);
      for (let j = 0; j < N; j++) {
        const current = C.at2D(i, j);
        C.set2D(i, j, current + a_ik * B.at2D(k, j));
      }
    }
  }
}

// Linear Layer / Projection: y = W * x + b
export function linear(
  W: TensorView,
  x: TensorView,
  b: TensorView,
  y: TensorView
): void {
  if (W.ndim !== 2 || x.ndim !== 1 || b.ndim !== 1 || y.ndim !== 1) {
    throw new Error("Dimension mismatch in linear layer");
  }

  const M = W.dim(0);
  const K = W.dim(1);

  if (x.dim(0) !== K || b.dim(0) !== M || y.dim(0) !== M) {
    throw new Error("Shape mismatch in linear parameters");
  }

  for (let i = 0; i < M; i++) {
    let sum = b.at1D(i);
    for (let j = 0; j < K; j++) {
      sum += W.at2D(i, j) * x.at1D(j);
    }
    y.set1D(i, sum);
  }
}

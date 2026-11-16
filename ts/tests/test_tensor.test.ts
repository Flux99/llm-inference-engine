import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { TensorShape } from "../src/tensor_shape.ts";
import { Tensor } from "../src/tensor.ts";
import { linear, matmul, relu } from "../src/ops_basic.ts";

describe("TensorShape Tests", () => {
  it("computes row-major strides correctly", () => {
    // 1D: [10] -> strides: [1]
    const shape1D = new TensorShape([10]);
    assert.strictEqual(shape1D.stride(0), 1);
    assert.strictEqual(shape1D.numel, 10);

    // 2D: [2, 3] -> strides: [3, 1]
    const shape2D = new TensorShape([2, 3]);
    assert.strictEqual(shape2D.stride(0), 3);
    assert.strictEqual(shape2D.stride(1), 1);
    assert.strictEqual(shape2D.numel, 6);

    // 4D (LLM tensor): [1, 8, 32, 64]
    const shape4D = new TensorShape([1, 8, 32, 64]);
    assert.strictEqual(shape4D.stride(0), 16384);
    assert.strictEqual(shape4D.stride(1), 2048);
    assert.strictEqual(shape4D.stride(2), 64);
    assert.strictEqual(shape4D.stride(3), 1);
    assert.strictEqual(shape4D.numel, 16384);
  });

  it("calculates flat memory offsets", () => {
    const shape = new TensorShape([2, 3]);
    assert.strictEqual(shape.offset(0, 0), 0);
    assert.strictEqual(shape.offset(0, 2), 2);
    assert.strictEqual(shape.offset(1, 0), 3);
    assert.strictEqual(shape.offset(1, 1), 4);
    assert.strictEqual(shape.offset(1, 2), 5);

    assert.throws(() => shape.offset(2, 0), RangeError);
    assert.throws(() => shape.offset(0, 3), RangeError);
  });
});

describe("Tensor & Zero-Copy Slicing Tests", () => {
  it("stores and mutates values", () => {
    const t = Tensor.fromData(new TensorShape([2, 3]), [1, 2, 3, 4, 5, 6]);
    assert.strictEqual(t.at2D(0, 0), 1);
    assert.strictEqual(t.at2D(1, 1), 5);
    t.set2D(1, 1, 99);
    assert.strictEqual(t.at2D(1, 1), 99);
  });

  it("proves that mutating a slice mutates the original parent tensor (zero-copy)", () => {
    const t = Tensor.fromData(new TensorShape([4, 2]), [
      1, 2,
      3, 4,
      5, 6,
      7, 8
    ]);

    const view = t.view();
    const slice = view.sliceDim0(1, 3); // Rows 1 to 2 -> shape [2, 2]

    assert.strictEqual(slice.dim(0), 2);
    assert.strictEqual(slice.dim(1), 2);
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

  it("computes 2D matmul: C = A * B", () => {
    const A = Tensor.fromData(new TensorShape([2, 3]), [
      1, 2, 3,
      4, 5, 6
    ]);
    const B = Tensor.fromData(new TensorShape([3, 2]), [
      7, 8,
      9, 1,
      2, 3
    ]);
    const C = Tensor.zeros(new TensorShape([2, 2]));

    matmul(A.view(), B.view(), C.view());

    assert.strictEqual(C.at2D(0, 0), 31);
    assert.strictEqual(C.at2D(0, 1), 19);
    assert.strictEqual(C.at2D(1, 0), 85);
    assert.strictEqual(C.at2D(1, 1), 55);
  });

  it("computes elementwise ReLU", () => {
    const input = Tensor.fromData(new TensorShape([4]), [-2.5, 0.0, 1.5, -0.1]);
    const output = Tensor.zeros(new TensorShape([4]));

    relu(input.view(), output.view());

    assert.strictEqual(output.data[0], 0);
    assert.strictEqual(output.data[1], 0);
    assert.strictEqual(output.data[2], 1.5);
    assert.strictEqual(output.data[3], 0);
  });
});

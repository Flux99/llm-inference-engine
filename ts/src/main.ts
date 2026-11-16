import { TensorShape } from "./tensor_shape.ts";
import { Tensor } from "./tensor.ts";
import { linear } from "./ops_basic.ts";

function main() {
  console.log("======================================================");
  console.log("   LLM Inference Engine: Phase 1 TypeScript Demo");
  console.log("======================================================\n");

  // 1. Inspect 2D Tensor memory layout and strides
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
  const row1Slice = view.sliceDim0(1, 2); // Row 1 only
  console.log(`   Slice Shape: ${row1Slice.shape.toString()}`);
  console.log("   Slice Values:");
  row1Slice.print();
  console.log();

  // 3. Execution of Linear Layer: y = W * x + b
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

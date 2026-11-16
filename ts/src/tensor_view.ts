import { TensorShape } from "./tensor_shape.ts";

/**
 * TensorView: Non-owning view over contiguous memory (Float32Array).
 * Provides zero-copy slicing along outer dimensions.
 */
export class TensorView {
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

  at1D(i: number): number {
    return this.data[i * this.shape.stride(0)];
  }

  set1D(i: number, val: number): void {
    this.data[i * this.shape.stride(0)] = val;
  }

  at2D(r: number, c: number): number {
    return this.data[r * this.shape.stride(0) + c * this.shape.stride(1)];
  }

  set2D(r: number, c: number, val: number): void {
    this.data[r * this.shape.stride(0) + c * this.shape.stride(1)] = val;
  }

  at(...indices: number[]): number {
    return this.data[this.shape.offset(...indices)];
  }

  // Zero-copy slice: Float32Array.subarray creates a view over the same buffer!
  sliceDim0(start: number, end: number): TensorView {
    if (start >= end || end > this.shape.dim(0)) {
      throw new RangeError("Invalid slice bounds");
    }

    const newDims = [...this.shape.dims];
    newDims[0] = end - start;

    const startOffset = start * this.shape.stride(0);
    const endOffset = startOffset + (end - start) * this.shape.stride(0);

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

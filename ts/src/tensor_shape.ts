/**
 * TensorShape: Multi-dimensional dimensions, contiguous row-major strides, and offset math.
 */
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

    // 1. Total elements = product of all dimensions
    this.numel = this.dims.reduce((acc, d) => acc * d, 1);

    // 2. Row-major contiguous strides (right to left):
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
    if (i < 0 || i >= this.dims.length) {
      throw new RangeError("Dimension index out of bounds");
    }
    return this.dims[i];
  }

  stride(i: number): number {
    if (i < 0 || i >= this.strides.length) {
      throw new RangeError("Stride index out of bounds");
    }
    return this.strides[i];
  }

  // Maps multi-dimensional coordinates (r, c, ...) to flat 1D memory offset
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

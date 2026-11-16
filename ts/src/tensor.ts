import { TensorShape } from "./tensor_shape.ts";
import { TensorView } from "./tensor_view.ts";

/**
 * Tensor: Owning container holding a contiguous Float32Array memory buffer.
 */
export class Tensor {
  readonly shape: TensorShape;
  private readonly data_: Float32Array;

  constructor(shape: TensorShape, fillValue: number = 0) {
    this.shape = shape;
    this.data_ = new Float32Array(this.shape.numel);
    if (fillValue !== 0) {
      this.data_.fill(fillValue);
    }
  }

  static zeros(shape: TensorShape): Tensor {
    return new Tensor(shape, 0);
  }

  static ones(shape: TensorShape): Tensor {
    return new Tensor(shape, 1);
  }

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

  // Returns a non-owning TensorView over this tensor's memory
  view(): TensorView {
    return new TensorView(this.data_, this.shape);
  }

  print(): void {
    this.view().print();
  }
}

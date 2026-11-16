/**
 * Data Types and Byte Sizes
 */
export enum DType {
  Float32 = "Float32",
  Float16 = "Float16",
  Int8 = "Int8",
  Int32 = "Int32",
}

export function dtypeSize(dtype: DType): number {
  switch (dtype) {
    case DType.Float32: return 4;
    case DType.Float16: return 2;
    case DType.Int8:    return 1;
    case DType.Int32:   return 4;
    default:
      throw new Error(`Unknown DType: ${dtype}`);
  }
}

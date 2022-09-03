#include <stdint.cuh>
#include <stdkernel.cuh>

/**
 * Broadcasting tensor multiply kernel.
 * @param a the flat contents of tensor A
 * @param nA number of elements in tensor A
 * @param b the flat contents of tensor B
 * @param nB number of elements in tensor B
 * @param out the flat contents where the result will be stored
 * @param nOut number of elements in the output tensor
 */
template <typename A, typename B, typename C>
KERNEL_TEMPLATE void multiply(A *a, size_t nA, B *b, size_t nB, C *out, size_t nOut) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nOut) {
        return;
    }
    out[i] = a[i % nA] * b[i % nB];
}

/* ALL TYPE PERMUTATIONS */
KERNEL_EXPORT void multiply_i8i8(int8_t *a, size_t nA, int8_t *b, size_t nB, int8_t *out, size_t nOut) { multiply<int8_t, int8_t, int8_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8u8(int8_t *a, size_t nA, uint8_t *b, size_t nB, int8_t *out, size_t nOut) { multiply<int8_t, uint8_t, int8_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8i16(int8_t *a, size_t nA, int16_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<int8_t, int16_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8u16(int8_t *a, size_t nA, uint16_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<int8_t, uint16_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8i32(int8_t *a, size_t nA, int32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int8_t, int32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8u32(int8_t *a, size_t nA, uint32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int8_t, uint32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8i64(int8_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int8_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8u64(int8_t *a, size_t nA, uint64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int8_t, uint64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8f32(int8_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<int8_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i8f64(int8_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<int8_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8i8(uint8_t *a, size_t nA, int8_t *b, size_t nB, int8_t *out, size_t nOut) { multiply<uint8_t, int8_t, int8_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8u8(uint8_t *a, size_t nA, uint8_t *b, size_t nB, uint8_t *out, size_t nOut) { multiply<uint8_t, uint8_t, uint8_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8i16(uint8_t *a, size_t nA, int16_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<uint8_t, int16_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8u16(uint8_t *a, size_t nA, uint16_t *b, size_t nB, uint16_t *out, size_t nOut) { multiply<uint8_t, uint16_t, uint16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8i32(uint8_t *a, size_t nA, int32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint8_t, int32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8u32(uint8_t *a, size_t nA, uint32_t *b, size_t nB, uint32_t *out, size_t nOut) { multiply<uint8_t, uint32_t, uint32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8i64(uint8_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint8_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8u64(uint8_t *a, size_t nA, uint64_t *b, size_t nB, uint64_t *out, size_t nOut) { multiply<uint8_t, uint64_t, uint64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8f32(uint8_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<uint8_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u8f64(uint8_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<uint8_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16i8(int16_t *a, size_t nA, int8_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<int16_t, int8_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16u8(int16_t *a, size_t nA, uint8_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<int16_t, uint8_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16i16(int16_t *a, size_t nA, int16_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<int16_t, int16_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16u16(int16_t *a, size_t nA, uint16_t *b, size_t nB, int16_t *out, size_t nOut) { multiply<int16_t, uint16_t, int16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16i32(int16_t *a, size_t nA, int32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int16_t, int32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16u32(int16_t *a, size_t nA, uint32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int16_t, uint32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16i64(int16_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int16_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16u64(int16_t *a, size_t nA, uint64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int16_t, uint64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16f32(int16_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<int16_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i16f64(int16_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<int16_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16i8(uint16_t *a, size_t nA, int8_t *b, size_t nB, uint16_t *out, size_t nOut) { multiply<uint16_t, int8_t, uint16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16u8(uint16_t *a, size_t nA, uint8_t *b, size_t nB, uint16_t *out, size_t nOut) { multiply<uint16_t, uint8_t, uint16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16i16(uint16_t *a, size_t nA, int16_t *b, size_t nB, uint16_t *out, size_t nOut) { multiply<uint16_t, int16_t, uint16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16u16(uint16_t *a, size_t nA, uint16_t *b, size_t nB, uint16_t *out, size_t nOut) { multiply<uint16_t, uint16_t, uint16_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16i32(uint16_t *a, size_t nA, int32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint16_t, int32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16u32(uint16_t *a, size_t nA, uint32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint16_t, uint32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16i64(uint16_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint16_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16u64(uint16_t *a, size_t nA, uint64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint16_t, uint64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16f32(uint16_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<uint16_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u16f64(uint16_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<uint16_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32i8(int32_t *a, size_t nA, int8_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int32_t, int8_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32u8(int32_t *a, size_t nA, uint8_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int32_t, uint8_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32i16(int32_t *a, size_t nA, int16_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int32_t, int16_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32u16(int32_t *a, size_t nA, uint16_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int32_t, uint16_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32u32(int32_t *a, size_t nA, uint32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<int32_t, uint32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32i64(int32_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int32_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32u64(int32_t *a, size_t nA, uint64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int32_t, uint64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32f32(int32_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<int32_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i32f64(int32_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<int32_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32i8(uint32_t *a, size_t nA, int8_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint32_t, int8_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32u8(uint32_t *a, size_t nA, uint8_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint32_t, uint8_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32i16(uint32_t *a, size_t nA, int16_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint32_t, int16_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32u16(uint32_t *a, size_t nA, uint16_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint32_t, uint16_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32i32(uint32_t *a, size_t nA, int32_t *b, size_t nB, int32_t *out, size_t nOut) { multiply<uint32_t, int32_t, int32_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32i64(uint32_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint32_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32u64(uint32_t *a, size_t nA, uint64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint32_t, uint64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32f32(uint32_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<uint32_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u32f64(uint32_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<uint32_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64i8(int64_t *a, size_t nA, int8_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, int8_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64u8(int64_t *a, size_t nA, uint8_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, uint8_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64i16(int64_t *a, size_t nA, int16_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, int16_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64u16(int64_t *a, size_t nA, uint16_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, uint16_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64i32(int64_t *a, size_t nA, int32_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, int32_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64u32(int64_t *a, size_t nA, uint32_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, uint32_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64u64(int64_t *a, size_t nA, uint64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<int64_t, uint64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64f32(int64_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<int64_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_i64f64(int64_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<int64_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64i8(uint64_t *a, size_t nA, int8_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, int8_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64u8(uint64_t *a, size_t nA, uint8_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, uint8_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64i16(uint64_t *a, size_t nA, int16_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, int16_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64u16(uint64_t *a, size_t nA, uint16_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, uint16_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64i32(uint64_t *a, size_t nA, int32_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, int32_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64u32(uint64_t *a, size_t nA, uint32_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, uint32_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64i64(uint64_t *a, size_t nA, int64_t *b, size_t nB, int64_t *out, size_t nOut) { multiply<uint64_t, int64_t, int64_t>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64f32(uint64_t *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<uint64_t, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_u64f64(uint64_t *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<uint64_t, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32i8(float *a, size_t nA, int8_t *b, size_t nB, float *out, size_t nOut) { multiply<float, int8_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32u8(float *a, size_t nA, uint8_t *b, size_t nB, float *out, size_t nOut) { multiply<float, uint8_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32i16(float *a, size_t nA, int16_t *b, size_t nB, float *out, size_t nOut) { multiply<float, int16_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32u16(float *a, size_t nA, uint16_t *b, size_t nB, float *out, size_t nOut) { multiply<float, uint16_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32i32(float *a, size_t nA, int32_t *b, size_t nB, float *out, size_t nOut) { multiply<float, int32_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32u32(float *a, size_t nA, uint32_t *b, size_t nB, float *out, size_t nOut) { multiply<float, uint32_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32i64(float *a, size_t nA, int64_t *b, size_t nB, float *out, size_t nOut) { multiply<float, int64_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32u64(float *a, size_t nA, uint64_t *b, size_t nB, float *out, size_t nOut) { multiply<float, uint64_t, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32f32(float *a, size_t nA, float *b, size_t nB, float *out, size_t nOut) { multiply<float, float, float>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f32f64(float *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<float, double, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64i8(double *a, size_t nA, int8_t *b, size_t nB, double *out, size_t nOut) { multiply<double, int8_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64u8(double *a, size_t nA, uint8_t *b, size_t nB, double *out, size_t nOut) { multiply<double, uint8_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64i16(double *a, size_t nA, int16_t *b, size_t nB, double *out, size_t nOut) { multiply<double, int16_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64u16(double *a, size_t nA, uint16_t *b, size_t nB, double *out, size_t nOut) { multiply<double, uint16_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64i32(double *a, size_t nA, int32_t *b, size_t nB, double *out, size_t nOut) { multiply<double, int32_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64u32(double *a, size_t nA, uint32_t *b, size_t nB, double *out, size_t nOut) { multiply<double, uint32_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64i64(double *a, size_t nA, int64_t *b, size_t nB, double *out, size_t nOut) { multiply<double, int64_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64u64(double *a, size_t nA, uint64_t *b, size_t nB, double *out, size_t nOut) { multiply<double, uint64_t, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64f32(double *a, size_t nA, float *b, size_t nB, double *out, size_t nOut) { multiply<double, float, double>(a, nA, b, nB, out, nOut); }
KERNEL_EXPORT void multiply_f64f64(double *a, size_t nA, double *b, size_t nB, double *out, size_t nOut) { multiply<double, double, double>(a, nA, b, nB, out, nOut); }
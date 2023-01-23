#include <stdint.cuh>
#include <stdkernel.cuh>

template <typename A, typename B>
KERNEL_TEMPLATE void cast(A *a, B *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    b[i] = (B)a[i];
}

/* ALL TYPE PERMUTATIONS */
KERNEL_EXPORT void cast_i8i8(int8_t *a, int8_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i8i16(int8_t *a, int16_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i8i32(int8_t *a, int32_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i8i64(int8_t *a, int64_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i8f32(int8_t *a, float *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i8f64(int8_t *a, double *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i16i8(int16_t *a, int8_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i16i16(int16_t *a, int16_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i16i32(int16_t *a, int32_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i16i64(int16_t *a, int64_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i16f32(int16_t *a, float *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i16f64(int16_t *a, double *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i32i8(int32_t *a, int8_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i32i16(int32_t *a, int16_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i32i32(int32_t *a, int32_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i32i64(int32_t *a, int64_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i32f32(int32_t *a, float *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i32f64(int32_t *a, double *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i64i8(int64_t *a, int8_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i64i16(int64_t *a, int16_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i64i32(int64_t *a, int32_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i64i64(int64_t *a, int64_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i64f32(int64_t *a, float *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_i64f64(int64_t *a, double *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f32i8(float *a, int8_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f32i16(float *a, int16_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f32i32(float *a, int32_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f32i64(float *a, int64_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f32f32(float *a, float *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f32f64(float *a, double *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f64i8(double *a, int8_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f64i16(double *a, int16_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f64i32(double *a, int32_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f64i64(double *a, int64_t *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f64f32(double *a, float *b, size_t n) { cast(a, b, n); }
KERNEL_EXPORT void cast_f64f64(double *a, double *b, size_t n) { cast(a, b, n); }
#include <stdint.cuh>
#include <stdkernel.cuh>

template <typename T>
KERNEL_TEMPLATE void less_than(T *a, uint8_t *result, T value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    result[i] = a[i] < value;
}

// ALL DATA TYPES
KERNEL_EXPORT void less_than_i8(int8_t *a, uint8_t *result, int8_t value, size_t n) { less_than(a, result, value, n); }
KERNEL_EXPORT void less_than_i16(int16_t *a, uint8_t *result, int16_t value, size_t n) { less_than(a, result, value, n); }
KERNEL_EXPORT void less_than_i32(int32_t *a, uint8_t *result, int32_t value, size_t n) { less_than(a, result, value, n); }
KERNEL_EXPORT void less_than_i64(int64_t *a, uint8_t *result, int64_t value, size_t n) { less_than(a, result, value, n); }
KERNEL_EXPORT void less_than_f32(float *a, uint8_t *result, float value, size_t n) { less_than(a, result, value, n); }
KERNEL_EXPORT void less_than_f64(double *a, uint8_t *result, double value, size_t n) { less_than(a, result, value, n); }
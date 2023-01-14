#include <stdint.cuh>
#include <stdkernel.cuh>

// TODO: OPTIMIZE
template <typename A, typename B, typename C>
KERNEL_TEMPLATE void matmul(A *a, B *b, C *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) {
    size_t batchIdx = blockDim.z * blockIdx.z + threadIdx.z;

    if (batchIdx >= batchSize) return;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // row
    size_t j = blockIdx.y * blockDim.y + threadIdx.y; // col

    size_t idx = i * n + j;

    if (idx >= batchSize * m * n) return;

    i %= m;
    j %= n;


    if (!transposeA && !transposeB) {
        C sum = 0;
        for (size_t l = 0; l < k; l++) {
            sum += a[batchIdx * strideA + i * k + l] * b[batchIdx * strideB + l * n + j];
        }
        c[batchIdx * strideC + i * n + j] = sum;
    } else if (transposeA && !transposeB) {
        C sum = 0;
        for (size_t l = 0; l < k; l++) {
            sum += a[batchIdx * strideA + l * m + i] * b[batchIdx * strideB + l * n + j];
        }
        c[batchIdx * strideC + i * n + j] = sum;
    } else if (!transposeA && transposeB) {
        C sum = 0;
        for (size_t l = 0; l < k; l++) {
            sum += a[batchIdx * strideA + i * k + l] * b[batchIdx * strideB + j * k + l];
        }
        c[batchIdx * strideC + i * n + j] = sum;
    } else {
        C sum = 0;
        for (size_t l = 0; l < k; l++) {
            sum += a[batchIdx * strideA + l * m + i] * b[batchIdx * strideB + j * k + l];
        }
        c[batchIdx * strideC + i * n + j] = sum;
    }
}


/* ALL TYPE PERMUTATIONS */
KERNEL_EXPORT void matmul_i8i8(int8_t *a, int8_t *b, int8_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int8_t, int8_t, int8_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i8i16(int8_t *a, int16_t *b, int16_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int8_t, int16_t, int16_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i8i32(int8_t *a, int32_t *b, int32_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int8_t, int32_t, int32_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i8i64(int8_t *a, int64_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int8_t, int64_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i8f32(int8_t *a, float *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int8_t, float, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i8f64(int8_t *a, double *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int8_t, double, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i16i8(int16_t *a, int8_t *b, int16_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int16_t, int8_t, int16_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i16i16(int16_t *a, int16_t *b, int16_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int16_t, int16_t, int16_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i16i32(int16_t *a, int32_t *b, int32_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int16_t, int32_t, int32_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i16i64(int16_t *a, int64_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int16_t, int64_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i16f32(int16_t *a, float *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int16_t, float, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i16f64(int16_t *a, double *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int16_t, double, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i32i8(int32_t *a, int8_t *b, int32_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int32_t, int8_t, int32_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i32i16(int32_t *a, int16_t *b, int32_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int32_t, int16_t, int32_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i32i32(int32_t *a, int32_t *b, int32_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int32_t, int32_t, int32_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i32i64(int32_t *a, int64_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int32_t, int64_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i32f32(int32_t *a, float *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int32_t, float, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i32f64(int32_t *a, double *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int32_t, double, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i64i8(int64_t *a, int8_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int64_t, int8_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i64i16(int64_t *a, int16_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int64_t, int16_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i64i32(int64_t *a, int32_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int64_t, int32_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i64i64(int64_t *a, int64_t *b, int64_t *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int64_t, int64_t, int64_t>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i64f32(int64_t *a, float *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int64_t, float, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_i64f64(int64_t *a, double *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<int64_t, double, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f32i8(float *a, int8_t *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<float, int8_t, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f32i16(float *a, int16_t *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<float, int16_t, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f32i32(float *a, int32_t *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<float, int32_t, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f32i64(float *a, int64_t *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<float, int64_t, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f32f32(float *a, float *b, float *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<float, float, float>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f32f64(float *a, double *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<float, double, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f64i8(double *a, int8_t *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<double, int8_t, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f64i16(double *a, int16_t *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<double, int16_t, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f64i32(double *a, int32_t *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<double, int32_t, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f64i64(double *a, int64_t *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<double, int64_t, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f64f32(double *a, float *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<double, float, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
KERNEL_EXPORT void matmul_f64f64(double *a, double *b, double *c, size_t m, size_t n, size_t k, uint32_t transposeA, uint32_t transposeB, size_t strideA, size_t strideB, size_t strideC, size_t batchSize) { matmul<double, double, double>(a, b, c, m, n, k, transposeA, transposeB, strideA, strideB, strideC, batchSize); }
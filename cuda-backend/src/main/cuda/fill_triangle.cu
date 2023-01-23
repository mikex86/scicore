#include <stdint.cuh>
#include <stdkernel.cuh>

template <typename T>
KERNEL_TEMPLATE void fill_triangle(T *data, int numBatches, int width, int height, T topValue, T bottomValue) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t batch = blockIdx.z;

    if (i >= width || j >= height || batch >= numBatches) {
        return;
    }

    if (i < j) {
        data[batch * width * height + j * width + i] = bottomValue;
    } else {
        data[batch * width * height + j * width + i] = topValue;
    }
}

// ALL DATA TYPES
KERNEL_EXPORT void fill_triangle_i8(int8_t *data, int numBatches, int width, int height, int8_t topValue, int8_t bottomValue) { fill_triangle(data, numBatches, width, height, topValue, bottomValue); }
KERNEL_EXPORT void fill_triangle_i16(int16_t *data, int numBatches, int width, int height, int16_t topValue, int16_t bottomValue) { fill_triangle(data, numBatches, width, height, topValue, bottomValue); }
KERNEL_EXPORT void fill_triangle_i32(int32_t *data, int numBatches, int width, int height, int32_t topValue, int32_t bottomValue) { fill_triangle(data, numBatches, width, height, topValue, bottomValue); }
KERNEL_EXPORT void fill_triangle_i64(int64_t *data, int numBatches, int width, int height, int64_t topValue, int64_t bottomValue) { fill_triangle(data, numBatches, width, height, topValue, bottomValue); }
KERNEL_EXPORT void fill_triangle_f32(float *data, int numBatches, int width, int height, float topValue, float bottomValue) { fill_triangle(data, numBatches, width, height, topValue, bottomValue); }
KERNEL_EXPORT void fill_triangle_f64(double *data, int numBatches, int width, int height, double topValue, double bottomValue) { fill_triangle(data, numBatches, width, height, topValue, bottomValue); }
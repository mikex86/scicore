#include "tanh.h"
#include "optimize.h"
#include <cmath>

template<typename T>
void tblas_tanh(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = std::tanh(in[i]);
    }
}

template<typename T>
void tblas_tanh_gradients(const T *tanhResulIn, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = 1 - (tanhResulIn[i] * tanhResulIn[i]);
    }
}

// check SVML is available
// This is mostly compiler damage control, as most compilers will automatically vectorize
// the std::tanh call.
#if defined(__AVX__) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
#include "vectorize_avx.h"
unary_op_nd(tanh, float, _mm256_tanh_ps, std::tanh);
unary_op_hook_optimizations(
        tanh, float,
        {
            tblas_tanh_nd(in, out, nElements);
        },
        {
        }
);
template<>
void tblas_tanh_gradients(const float *tanhResulIn, float* out, size_t nElements) {
    size_t i = 0;
    if (nElements >= 8) {
        size_t vectorizeEndIdx = nElements - 8;
        __m256 oneConst = _mm256_set1_ps(1.0f);
        for (; i < vectorizeEndIdx; i += 8) {
            __m256 inVec = _mm256_loadu_ps(tanhResulIn + i);
            __m256 outVec = _mm256_sub_ps(oneConst, _mm256_mul_ps(inVec, inVec));
            _mm256_storeu_ps(out + i, outVec);
        }
    }
    for (; i < nElements; i++) {
        out[i] = 1 - (tanhResulIn[i] * tanhResulIn[i]);
    }
}
#endif
#if __ARM_NEON__
// Arm Neon specific implementation
#include "vectorize_armneon.h"
#include <neon_mathfun.h>

float32x4_t tanh_ps(float32x4_t x) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    float32x4_t exp2x = exp_ps(vaddq_f32(x, x));
    float32x4_t exp2x_plus_1 = vaddq_f32(exp2x, vdupq_n_f32(1.0f));
    float32x4_t exp2x_minus_1 = vsubq_f32(exp2x, vdupq_n_f32(1.0f));
    return vdivq_f32(exp2x_minus_1, exp2x_plus_1);
}

unary_op_nd(tanh, float, tanh_ps, std::tanh);

unary_op_hook_optimizations(
        tanh, float,
        {
            tblas_tanh_nd(in, out, nElements);
        },
        {
        }
);

template<>
void tblas_tanh_gradients(const float *tanhResulIn, float *out, size_t nElements) {
    size_t i = 0;
    if (nElements >= 4) {
        size_t vectorizeEndIdx = nElements - 4;
        float32x4_t oneConst = vdupq_n_f32(1.0f);
        for (; i < vectorizeEndIdx; i += 4) {
            float32x4_t inVec = vld1q_f32(tanhResulIn + i);
            float32x4_t outVec = vsubq_f32(oneConst, vmulq_f32(inVec, inVec));
            vst1q_f32(out + i, outVec);
        }
    }
    for (; i < nElements; i++) {
        out[i] = 1 - (tanhResulIn[i] * tanhResulIn[i]);
    }
}

#endif

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_tanh);
UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_tanh_gradients);

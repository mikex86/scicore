#include "sigmoid.h"
#include "optimize.h"
#include <cmath>

static float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

// TODO: CHECK IF MSVC VECTORIZES THE NAIVE IMPLEMENTATION
template<typename T>
void tblas_sigmoid(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

// check SVML is available
#if defined(__AVX__) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
#include "vectorize_avx.h"

__m256 sigmoid_ps(__m256 x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    __m256 minus_x = _mm256_sub_ps(_mm256_set1_ps(0.0f), x);
    __m256 exp_x = _mm256_exp_ps(minus_x);
    __m256 exp_x_plus_1 = _mm256_add_ps(exp_x, _mm256_set1_ps(1.0f));
    return _mm256_div_ps(_mm256_set1_ps(1.0f), exp_x_plus_1);
}

unary_op_nd(sigmoid, float, sigmoid_ps, sigmoid);
unary_op_hook_optimizations(
        sigmoid, float,
        {
            tblas_sigmoid_nd(in, out, nElements);
        },
        {
        }
);
#endif
#ifdef __ARM_NEON__
// Arm Neon specific implementation
#include "vectorize_armneon.h"
#include <neon_mathfun.h>

float32x4_t sigmoid_ps(float32x4_t x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    float32x4_t minus_x = vnegq_f32(x);
    float32x4_t exp_x = exp_ps(minus_x);
    float32x4_t exp_x_plus_1 = vaddq_f32(exp_x, vdupq_n_f32(1.0f));
    return vdivq_f32(vdupq_n_f32(1.0f), exp_x_plus_1);
}
unary_op_nd(sigmoid, float, sigmoid_ps, sigmoid);
unary_op_hook_optimizations(
        sigmoid, float,
        {
            tblas_sigmoid_nd(in, out, nElements);
        },
        {
        }
);
#endif


UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_sigmoid);

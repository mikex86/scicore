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
void tblas_tanh_gradients(const T *tanhResulIn, T* out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = 1 - (tanhResulIn[i] * tanhResulIn[i]);
    }
}

// AVX specific implementation
#ifdef __AVX__

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
    size_t vectorizeEndIdx = nElements - 8;
    size_t i = 0;
    __m256 oneConst = _mm256_set1_ps(1.0f);
    for (; i < vectorizeEndIdx; i += 8) {
        __m256 inVec = _mm256_loadu_ps(tanhResulIn + i);
        __m256 outVec = _mm256_sub_ps(oneConst, _mm256_mul_ps(inVec, inVec));
        _mm256_storeu_ps(out + i, outVec);
    }
    for (; i < nElements; i++) {
        out[i] = 1 - (tanhResulIn[i] * tanhResulIn[i]);
    }
}

#endif

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_tanh);
UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_tanh_gradients);
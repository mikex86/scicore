#include "exp.h"
#include "optimize.h"
#include <cmath>

template<typename T>
void tblas_exp(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = std::exp(in[i]);
    }
}

// AVX specific implementation
#ifdef __AVX2__
#include "vectorize_avx.h"
unary_op_nd(exp, float, _mm256_exp_ps, std::exp);
unary_op_hook_optimizations(
        exp, float,
{
tblas_exp_nd(in, out, nElements);
},
{
}
);
#endif


UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_exp)

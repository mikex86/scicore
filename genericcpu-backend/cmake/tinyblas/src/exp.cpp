#include "exp.h"
#include "optimize.h"
#include <cmath>

template<typename T>
void tblas_exp(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = std::exp(in[i]);
    }
}

// check SVML is available
// This is mostly compiler damage control, as most compilers will automatically vectorize
// the std::exp call. GCC for example will use libmvec, which is an SVML-like library that is also
// open source.
#if defined(__AVX__) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
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

#include "log.h"
#include "optimize.h"
#include <cmath>

template<typename T>
void tblas_log(const T *in, T *out, size_t nElements) {
    for (size_t i = 0; i < nElements; i++) {
        out[i] = std::log(in[i]);
    }
}


// check SVML is available
// This is mostly compiler damage control, as most compilers will automatically vectorize
// the std::log call.
#if defined(__AVX__) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))

#include "vectorize_avx.h"
unary_op_nd(log, float, _mm256_log_ps, std::log);
unary_op_hook_optimizations(
        log, float,
        {
            tblas_log_nd(in, out, nElements);
        },
        {
        }
);
#endif

UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_log);

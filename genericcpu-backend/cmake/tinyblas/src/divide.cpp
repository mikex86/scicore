#include "divide.h"
#include "shapeutils.h"
#include <cstring>

template<typename A, typename B, typename C>
void tblas_divide(const A *a, const B *b, C *c,
                 const size_t *shapeA, const size_t *stridesA, size_t nDimsA,
                 const size_t *shapeB, const size_t *stridesB, size_t nDimsB,
                 const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
        size_t cIndexFlat = getFlatIndex(outputIndex, stridesC, nDimsC);
        c[cIndexFlat] = a[aIndexFlat] / b[bIndexFlat];
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}

#ifdef __AVX__

#include "vectorize_avx.h"
nd_by_scalar_op(divide, float, _mm256_div_ps, /)
nd_by_nd_op(divide, float, _mm256_div_ps, /);

op_hook_optimizations(
        divide, float,
        {
            if (tblas_divide_nd_by_scalar(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB,
                                         shapeC, stridesC, nDimsC))
                return;
            if (tblas_divide_nd_by_nd(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB,
                                     shapeC, stridesC, nDimsC))
                return;
        },
        {
            auto *outputIndex = new size_t[nDimsC];
            memset(outputIndex, 0, sizeof(size_t) * nDimsC);
            do {
                size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA,
                                                            nDimsC);
                size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB,
                                                            nDimsC);
                size_t cIndexFlat = getFlatIndex(outputIndex, stridesC, nDimsC);
                c[cIndexFlat] = a[aIndexFlat] / b[bIndexFlat];
            } while (incrementIndex(outputIndex, shapeC, nDimsC));
            delete[] outputIndex;
        }
);
#endif

BINARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_divide)
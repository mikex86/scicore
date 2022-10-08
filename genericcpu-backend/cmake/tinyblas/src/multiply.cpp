#include "multiply.h"
#include "shapeutils.h"
#include <cstring>

template<typename A, typename B, typename C>
void tblas_multiply(const A *a, const B *b, C *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeB,
                    size_t *stridesB, size_t nDimsB,
                    size_t *shapeC, size_t *, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    size_t cIndexFlat = 0;
    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
        c[cIndexFlat] = a[aIndexFlat] * b[bIndexFlat];
        cIndexFlat++;
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}

OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_multiply)
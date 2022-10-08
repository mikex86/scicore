#pragma once

#include <cassert>
#include "forceinline.h"

FORCE_INLINE bool incrementIndex(size_t *outputIndex, const size_t *shape, size_t nDims) {
    for (int i = (long) nDims - 1; i >= 0; i--) {
        if (outputIndex[i] < shape[i] - 1) {
            outputIndex[i]++;
            return true;
        } else {
            outputIndex[i] = 0;
        }
    }
    return false;
}

FORCE_INLINE size_t
getFlatIndexConstrained(const size_t *outputIndex, const size_t *shape, const size_t *strides, size_t nDims, size_t nDimsOut) {
    assert(nDims <= nDimsOut);
    size_t nNewDims = nDimsOut - nDims;
    size_t flatIndex = 0;
    for (size_t dim = 0; dim < nDims; dim++) {
        size_t stride = strides[dim];
        flatIndex += (outputIndex[dim + nNewDims] % shape[dim]) * stride;
    }
    return flatIndex;
}
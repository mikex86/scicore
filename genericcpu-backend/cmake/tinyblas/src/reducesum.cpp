#include "reducesum.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "shapeutils.h"

template<typename T>
void tblas_reducesum(const T *a, T *c,
                     size_t *shapeA, size_t *stridesA, size_t nDimsA,
                     size_t *shapeC, size_t *stridesC, size_t nDimsC,
                     int64_t dimension, bool keepDims) {
    if (dimension == -1) {
        // reduce all dimensions
        if (!keepDims) {
            if (!(nDimsC == 0 || (nDimsC == 1 && shapeC[0] == 1))) {
                throw std::runtime_error("tblas_reducesum: when reducing all dimensions and keepDims is false, the output shape must be a scalar");
            }
        } else {
            if (nDimsC != nDimsA) {
                throw std::runtime_error("tblas_reducesum: when reducing all dimensions and keepDims is true, the number of output dimensions must be the same as the number of input dimensions");
            }
            for (size_t i = 0; i < nDimsC; i++) {
                if (shapeC[i] != 1) {
                    throw std::runtime_error("tblas_reducesum: when reducing all dimensions and keepDims is true, the output shape must be 1 in all dimensions");
                }
            }
        }
        size_t nElementsA = 1;
        for (int i = 0; i < nDimsA; i++) {
            nElementsA *= shapeA[i];
        }
        for (size_t i = 0; i < nElementsA; i++) {
            c[0] += a[i];
        }
    } else {
        // reduce one dimension
        if (dimension < 0 || dimension >= nDimsA) {
            throw std::runtime_error("tblas_reducesum: dimension out of range");
        }
        // check n dims
        if (!keepDims) {
            if (nDimsC != nDimsA - 1) {
                throw std::runtime_error("tblas_reducesum: nDimsC != nDimsA - 1");
            }
        } else {
            if (nDimsC != nDimsA) {
                throw std::runtime_error("tblas_reducesum: nDimsC != nDimsA");
            }
        }
        // check shape if shapeC is expected reduced shape.
        // if keepDims is true, shapeC at dimension is expected to be 1.
        // if keepDims is false, dimension is expected to be removed.
        for (int i = 0, j = 0; i < nDimsA; i++) {
            if (i == dimension) {
                if (keepDims) {
                    if (shapeC[i + j] != 1) {
                        throw std::runtime_error("tblas_reducesum: shapeC[dimension] != 1 when keepDims=true");
                    }
                } else {
                    j++;
                }
            } else {
                if (shapeC[i - j] != shapeA[i]) {
                    throw std::runtime_error("tblas_reducesum: shapeC[i + j] != shapeA[i]");
                }
            }
        }
        auto *completeIndex = new size_t[nDimsA];
        memset(completeIndex, 0, sizeof(size_t) * nDimsA);
        auto *reducedIndex = new size_t[nDimsC];
        memset(reducedIndex, 0, sizeof(size_t) * nDimsC);
        while (true) {
            T sum = 0;
            for (size_t i = 0; i < shapeA[dimension]; i++) {
                completeIndex[dimension] = i;
                size_t aIndexFlat = getFlatIndex(completeIndex, stridesA, nDimsA);
                sum += a[aIndexFlat];
            }
            size_t cIndexFlat = getFlatIndex(reducedIndex, stridesC, nDimsC);
            c[cIndexFlat] = sum;
            if (!incrementIndex(reducedIndex, shapeC, nDimsC)) {
                break;
            }
            if (!incrementIndex(completeIndex, shapeA, nDimsA)) {
                break;
            }
        }
    }
}

void tblas_reducesum(const int8_t *a, int8_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeC,
                     size_t *stridesC, size_t nDimsC, int64_t dimension, bool keepDims) {
    tblas_reducesum<int8_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension, keepDims);
}

void tblas_reducesum(const int16_t *a, int16_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeC,
                     size_t *stridesC, size_t nDimsC, int64_t dimension, bool keepDims) {
    tblas_reducesum<int16_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension, keepDims);
}

void tblas_reducesum(const int32_t *a, int32_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeC,
                     size_t *stridesC, size_t nDimsC, int64_t dimension, bool keepDims) {
    tblas_reducesum<int32_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension, keepDims);
}

void tblas_reducesum(const int64_t *a, int64_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeC,
                     size_t *stridesC, size_t nDimsC, int64_t dimension, bool keepDims) {
    tblas_reducesum<int64_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension, keepDims);
}

void tblas_reducesum(const float *a, float *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeC,
                     size_t *stridesC, size_t nDimsC, int64_t dimension, bool keepDims) {
    tblas_reducesum<float>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension, keepDims);
}

void tblas_reducesum(const double *a, double *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeC,
                     size_t *stridesC, size_t nDimsC, int64_t dimension, bool keepDims) {
    tblas_reducesum<double>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension, keepDims);
}

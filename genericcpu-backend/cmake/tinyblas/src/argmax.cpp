#include <argmax.h>

template<typename T>
void tblas_argmax(const T *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    if (dimension == -1) {
        size_t nElementsA = 1;
        for (size_t i = 0; i < nDimsA; i++) {
            nElementsA *= shapeA[i];
        }
        size_t maxIndex = 0;
        T maxValue = a[0];
        for (size_t i = 1; i < nElementsA; i++) {
            if (a[i] > maxValue) {
                maxIndex = i;
                maxValue = a[i];
            }
        }
        c[0] = maxIndex;
    } else {
        size_t nElementsA = 1;
        for (size_t i = 0; i < nDimsA; i++) {
            nElementsA *= shapeA[i];
        }
        size_t nElementsC = 1;
        for (size_t i = 0; i < nDimsC; i++) {
            nElementsC *= shapeC[i];
        }
        size_t strideA = stridesA[dimension];
        size_t strideC = stridesC[dimension];
        size_t nElementsPerSlice = nElementsA / shapeA[dimension];
        for (size_t i = 0; i < nElementsC; i++) {
            size_t offsetA = i * strideC;
            size_t offsetC = i * strideC;
            size_t maxIndex = 0;
            T maxValue = a[offsetA];
            for (size_t j = 1; j < nElementsPerSlice; j++) {
                if (a[offsetA + j * strideA] > maxValue) {
                    maxIndex = j;
                    maxValue = a[offsetA + j * strideA];
                }
            }
            c[offsetC] = maxIndex;
        }
    }
}

void tblas_argmax(const int8_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    tblas_argmax<int8_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension);
}

void tblas_argmax(const int16_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    tblas_argmax<int16_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension);
}

void tblas_argmax(const int32_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    tblas_argmax<int32_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension);
}

void tblas_argmax(const int64_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    tblas_argmax<int64_t>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension);
}

void tblas_argmax(const float *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    tblas_argmax<float>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension);
}

void tblas_argmax(const double *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension) {
    tblas_argmax<double>(a, c, shapeA, stridesA, nDimsA, shapeC, stridesC, nDimsC, dimension);
}
#include <argmax.h>
#include <cstring>
#include "shapeutils.h"

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
        auto *aIndex = new size_t[nDimsA];
        memset(aIndex, 0, nDimsA * sizeof(size_t));
        auto *outputIndex = new size_t[nDimsC];
        memset(outputIndex, 0, sizeof(size_t) * nDimsC);

        do {
            size_t aFlatIndex = getFlatIndex(aIndex, stridesA, nDimsA);
            T maxValue = a[aFlatIndex];
            size_t maxIndex = aIndex[dimension];
            for (size_t i = 1; i < shapeA[dimension]; i++) {
                aIndex[dimension] = i;
                aFlatIndex = getFlatIndex(aIndex, stridesA, nDimsA);
                if (a[aFlatIndex] > maxValue) {
                    maxValue = a[aFlatIndex];
                    maxIndex = i;
                }
            }
            size_t cFlatIndex = getFlatIndex(outputIndex, stridesC, nDimsC);
            c[cFlatIndex] = maxIndex;
            aIndex[dimension] = 0;

            // increment aIndex on all dimensions except the dimension we're reducing
            for (size_t i = 0; i < nDimsA; i++) {
                if (i != dimension) {
                    aIndex[nDimsA - i - 1]++;
                    if (aIndex[nDimsA - i - 1] < shapeA[nDimsA - i - 1]) {
                        break;
                    } else {
                        aIndex[nDimsA - i - 1] = 0;
                    }
                }
            }
        } while (incrementIndex(outputIndex, shapeC, nDimsC));

        delete[] aIndex;
        delete[] outputIndex;
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
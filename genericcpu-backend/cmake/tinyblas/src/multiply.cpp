#include "multiply.h"
#include "shapeutils.h"
#include <cstring>

#ifndef __ARM_NEON__
template<typename A, typename B, typename C>
void tblas_multiply(const A *a, const B *b, C *c,
                    size_t *shapeA, size_t *stridesA,
                    size_t nDimsA, size_t *shapeB,
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
#else

#include <arm_neon.h>
#include <iostream>

template<typename A, typename B, typename C>
void tblas_multiply(const A *a, const B *b, C *c,
                    size_t *shapeA, size_t *stridesA,
                    size_t nDimsA, size_t *shapeB,
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

// returns true if optimization hits
bool tblas_multiply_nd_by_scalar(const float *a, const float *b, float *c,
                                 size_t *shapeA, size_t *stridesA,
                                 size_t nDimsA, size_t *shapeB,
                                 size_t *, size_t nDimsB,
                                 size_t *shapeC, size_t *stridesC, size_t nDimsC) {
    // if a has altered strides, return false
    if (!unalteredStrides(stridesA, shapeA, nDimsA)) {
        return false;
    }
    // if b is not a scalar, return false
    if (!(nDimsB == 0 || (nDimsB == 1 && shapeB[0] == 1))) {
        return false;
    }
    // if c is different shape than 'a', return false
    {
        if (nDimsA != nDimsC) {
            return false;
        }
        for (int i = 0; i < nDimsA; i++) {
            if (shapeA[i] != shapeC[i]) {
                return false;
            }
        }
    }
    // if c has altered strides, return false
    if (!unalteredStrides(stridesC, shapeC, nDimsC)) {
        return false;
    }

    size_t nElements = 1;
    for (int i = 0; i < nDimsA; i++) {
        nElements *= shapeA[i];
    }
    size_t nChunks = nElements / 4;
    size_t nRemainder = nElements % 4;
    float32x4_t scalar = vdupq_n_f32(*b);
    for (int i = 0; i < nChunks; i++) {
        float32x4_t aChunk = vld1q_f32(a);
        float32x4_t cChunk = vmulq_f32(aChunk, scalar);
        vst1q_f32(c, cChunk);
        a += 4;
        c += 4;
    }
    for (int i = 0; i < nRemainder; i++) {
        *c = *a * *b;
        a++;
        c++;
    }
    return true;
}


bool tblas_multiply_nd_by_nd(const float *a, const float *b, float *c,
                             size_t *shapeA, size_t *stridesA, size_t nDimsA,
                             size_t *shapeB, size_t *stridesB, size_t nDimsB,
                             size_t *shapeC, size_t *stridesC, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);
    if (nDimsC != nDimsA && nDimsC != nDimsB) {
//        std::cout << "nDimsC != nDimsA && nDimsC != nDimsB" << std::endl;
        return false;
    }
    // if shapeA != shapeC && shapeB != shapeC, return false
    bool shouldCheckB = true;
    if (nDimsC == nDimsA) {
        for (int i = 0; i < nDimsC; i++) {
            if (shapeA[i] != shapeC[i]) {
//                std::cout << "shapeA[i] != shapeC[i]" << std::endl;
                break;
            }
        }
        shouldCheckB = false;
    }
    if (shouldCheckB) {
        if (nDimsC == nDimsB) {
            for (int i = 0; i < nDimsC; i++) {
                if (shapeB[i] != shapeC[i]) {
//                    std::cout << "shapeB[i] != shapeC[i]" << std::endl;
                    return false;
                }
            }
        } else {
//            std::cout << "nDimsC != nDimsB" << std::endl;
            return false;
        }
    }
    if (!unalteredStrides(stridesC, shapeC, nDimsC)) {
//        std::cout << "!unalteredStrides(stridesC, shapeC, nDimsC)" << std::endl;
        return false;
    }
    size_t nElementsInLastDimA = shapeA[nDimsA - 1];
    size_t nElementsInLastDimB = shapeB[nDimsB - 1];
    if (stridesA[nDimsA - 1] != 1) {
//        std::cout << "stridesA[nDimsA - 1] != 1" << std::endl;
        return false;
    }
    if (stridesB[nDimsB - 1] != 1) {
//        std::cout << "stridesB[nDimsB - 1] != 1" << std::endl;
        return false;
    }
    if (!(nElementsInLastDimA == nElementsInLastDimB || nElementsInLastDimB == 1 || nElementsInLastDimA == 1)) {
//        std::cout << "!(nElementsInLastDimA == nElementsInLastDimB || nElementsInLastDimB == 1 || nElementsInLastDimA == 1)"
//                  << std::endl;
        return false;
    }
    size_t nElementsInLastDim = std::max(nElementsInLastDimA, nElementsInLastDimB);

    if (nElementsInLastDim < 4) {
//        std::cout << "nElementsInLastDim < 4" << std::endl;
        return false;
    }
    bool aIsScalarInLastDim = nElementsInLastDimA == 1;
    bool bIsScalarInLastDim = nElementsInLastDimB == 1;

    size_t nElementsC = 1;
    for (int i = 0; i < nDimsC; i++) {
        nElementsC *= shapeC[i];
    }
    size_t cFlatIndex = 0;
    size_t nChunks = nElementsInLastDim / 4;
    size_t nRemainder = nElementsInLastDim % 4;

    if (!aIsScalarInLastDim && !bIsScalarInLastDim) {
        do {
            size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
            size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
            for (int i = 0; i < nChunks; i++) {
                float32x4_t aVec = vld1q_f32(&a[aIndexFlat + i * 4]);
                float32x4_t bVec = vld1q_f32(&b[bIndexFlat + i * 4]);
                float32x4_t cVec = vmulq_f32(aVec, bVec);
                vst1q_f32(&c[cFlatIndex + i * 4], cVec);
            }
            for (int i = 0; i < nRemainder; i++) {
                c[cFlatIndex + nChunks * 4 + i] = a[aIndexFlat + nChunks * 4 + i] * b[bIndexFlat + nChunks * 4 + i];
            }
            cFlatIndex += nElementsInLastDim;
            outputIndex[nDimsC - 1] += nElementsInLastDim;
            incrementIndex(outputIndex, shapeC, nDimsC);
        } while (cFlatIndex < nElementsC);
    } else if (!aIsScalarInLastDim) {
        do {
            size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
            size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
            float32x4_t bScalar = vdupq_n_f32(b[bIndexFlat]);
            for (int i = 0; i < nChunks; i++) {
                float32x4_t aVec = vld1q_f32(&a[aIndexFlat + i * 4]);
                float32x4_t cVec = vmulq_f32(aVec, bScalar);
                vst1q_f32(&c[cFlatIndex + i * 4], cVec);
            }
            for (int i = 0; i < nRemainder; i++) {
                c[cFlatIndex + nChunks * 4 + i] = a[aIndexFlat + nChunks * 4 + i] * b[bIndexFlat];
            }
            cFlatIndex += nElementsInLastDim;
            outputIndex[nDimsC - 1] += nElementsInLastDim;
            incrementIndex(outputIndex, shapeC, nDimsC);
        } while (cFlatIndex < nElementsC);
    } else if (!bIsScalarInLastDim) {
        do {
            size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
            size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
            float32x4_t aScalar = vdupq_n_f32(a[aIndexFlat]);
            for (int i = 0; i < nChunks; i++) {
                float32x4_t bVec = vld1q_f32(&b[bIndexFlat + i * 4]);
                float32x4_t cVec = vmulq_f32(aScalar, bVec);
                vst1q_f32(&c[cFlatIndex + i * 4], cVec);
            }
            for (int i = 0; i < nRemainder; i++) {
                c[cFlatIndex + nChunks * 4 + i] = a[aIndexFlat] * b[bIndexFlat + nChunks * 4 + i];
            }
            cFlatIndex += nElementsInLastDim;
            outputIndex[nDimsC - 1] += nElementsInLastDim;
            incrementIndex(outputIndex, shapeC, nDimsC);
        } while (cFlatIndex < nElementsC);
    } else {
//        std::cout << "aIsScalarInLastDim && bIsScalarInLastDim" << std::endl;
        return false;
    }
    return true;
}

template<>
void tblas_multiply(const float *a, const float *b, float *c,
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,
                    size_t *shapeC, size_t *stridesC, size_t nDimsC) {

    // check for possible optimizations
    if (tblas_multiply_nd_by_scalar(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC,
                                    nDimsC)) {
        return;
    }
    if (tblas_multiply_nd_by_nd(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC,
                                nDimsC)) {
        return;
    }

    // fall back
//    std::cout << "fall back to naive implementation" << std::endl;

    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);

        size_t cIndexFlat = getFlatIndex(outputIndex, stridesC, nDimsC);
        c[cIndexFlat] = a[aIndexFlat] * b[bIndexFlat];
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}


#endif

BINARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_multiply)
#include "minus.h"
#include "shapeutils.h"
#include <cstring>

#ifndef __ARM_NEON__
template<typename A, typename B, typename C>
void tblas_minus(const A *a, const B *b, C *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeB,
                size_t *stridesB, size_t nDimsB,
                size_t *shapeC, size_t *, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    size_t cIndexFlat = 0;
    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
        c[cIndexFlat] = a[aIndexFlat] - b[bIndexFlat];
        cIndexFlat++;
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}
#else
#include <arm_neon.h>

template<typename A, typename B, typename C>
void tblas_minus(const A *a, const B *b, C *c, size_t *shapeA, size_t *stridesA, size_t nDimsA, size_t *shapeB,
                 size_t *stridesB, size_t nDimsB,
                 size_t *shapeC, size_t *, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    size_t cIndexFlat = 0;
    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
        c[cIndexFlat] = a[aIndexFlat] - b[bIndexFlat];
        cIndexFlat++;
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}

// returns true if optimization hits
bool tblas_minus_nd_by_scalar(const float *a, const float *b, float *c,
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
        float32x4_t cChunk = vsubq_f32(aChunk, scalar);
        vst1q_f32(c, cChunk);
        a += 4;
        c += 4;
    }
    for (int i = 0; i < nRemainder; i++) {
        *c = *a - *b;
        a++;
        c++;
    }
    return true;
}

bool tblas_minus_same_shape(const float *a, const float *b, float *c,
                               size_t *shapeA, size_t *stridesA,
                               size_t nDimsA, size_t *shapeB,
                               size_t *stridesB, size_t nDimsB,
                               size_t *shapeC, size_t *stridesC, size_t nDimsC) {
    // if a has altered strides, return false
    if (!unalteredStrides(stridesA, shapeA, nDimsA)) {
        return false;
    }
    // if b has altered strides, return false
    if (!unalteredStrides(stridesB, shapeB, nDimsB)) {
        return false;
    }
    // if b is different shape than 'a', return false
    {
        if (nDimsA != nDimsB) {
            return false;
        }
        for (int i = 0; i < nDimsA; i++) {
            if (shapeA[i] != shapeB[i]) {
                return false;
            }
        }
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
    for (int i = 0; i < nChunks; i++) {
        float32x4_t aChunk = vld1q_f32(a);
        float32x4_t bChunk = vld1q_f32(b);
        float32x4_t cChunk = vsubq_f32(aChunk, bChunk);
        vst1q_f32(c, cChunk);
        a += 4;
        b += 4;
        c += 4;
    }
    for (int i = 0; i < nRemainder; i++) {
        *c = *a - *b;
        a++;
        b++;
        c++;
    }
    return true;
}

template<>
void tblas_minus(const float *a, const float *b, float *c,
                size_t *shapeA, size_t *stridesA, size_t nDimsA,
                size_t *shapeB, size_t *stridesB, size_t nDimsB,
                size_t *shapeC, size_t *stridesC, size_t nDimsC) {

    // check for possible optimizations
    if (tblas_minus_nd_by_scalar(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC)) {
        return;
    }
    if (tblas_minus_same_shape(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC)) {
        return;
    }

    // fall back
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    size_t cIndexFlat = 0;
    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
        c[cIndexFlat] = a[aIndexFlat] - b[bIndexFlat];
        cIndexFlat++;
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}
#endif

BINARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_minus)
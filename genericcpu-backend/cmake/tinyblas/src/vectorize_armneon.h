#pragma once

#include <arm_neon.h>
#include <algorithm>

#define OPERANDS_SIZE 16

#define nd_by_scalar_op(op_name, type, vec_inst, scalar_op) \
bool tblas_##op_name##_nd_by_scalar(const type *a, const type *b, type *c,\
                                 const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                                 const size_t *shapeB, const size_t *, size_t nDimsB,\
                                 const size_t *shapeC, const size_t *stridesC, size_t nDimsC) { \
    size_t vecSize = OPERANDS_SIZE / sizeof(type); \
    /* if a has altered strides, return false */ \
    if (!unalteredStrides(stridesA, shapeA, nDimsA)) {\
        return false; \
    } \
    /* if c has altered strides, return false */ \
    if (!unalteredStrides(stridesC, shapeC, nDimsC)) {\
        return false;\
    }\
    /* if either 'a' or 'b' is a scalar, we continue. We return false, if both are scalars, or neither of them are scalars. */ \
    bool aIsScalar = nDimsA == 0 || (nDimsA == 1 && shapeA[0] == 1); \
    bool bIsScalar = nDimsB == 0 || (nDimsB == 1 && shapeB[0] == 1); \
    if (!(aIsScalar ^ bIsScalar)) {\
        return false;\
    } \
    size_t nElements = 1; \
    if (aIsScalar) {\
        for (int i = 0; i < nDimsB; i++) {\
            nElements *= shapeB[i];\
        }\
    } else { \
        for (int i = 0; i < nDimsA; i++) {\
            nElements *= shapeA[i];\
        }\
    }\
    size_t nChunks = nElements / vecSize;\
    size_t nRemainder = nElements % vecSize;\
    if (aIsScalar) {\
        float32x4_t scalar = vdupq_n_f32(*a);\
        for (int i = 0; i < nChunks; i++) {\
            float32x4_t bChunk = vld1q_f32(b);\
            float32x4_t cChunk = vec_inst(scalar, bChunk);\
            vst1q_f32(c, cChunk);\
            b += vecSize;\
            c += vecSize;\
        }\
        for (int i = 0; i < nRemainder; i++) {\
            *c = *a scalar_op *b;\
            b++;\
            c++;\
        }\
    } else {\
        float32x4_t scalar = vdupq_n_f32(*b);\
        for (int i = 0; i < nChunks; i++) {\
            float32x4_t aChunk = vld1q_f32(a);\
            float32x4_t cChunk = vec_inst(aChunk, scalar);\
            vst1q_f32(c, cChunk);\
            a += vecSize;\
            c += vecSize;\
        }\
        for (int i = 0; i < nRemainder; i++) {\
            *c = (*a) scalar_op (*b);\
            a++;\
            c++;\
        }\
    }\
    return true;\
}

#define nd_by_nd_op(op_name, type, vec_inst, scalar_op) \
bool tblas_##op_name##_nd_by_nd(const type *a, const type *b, type *c, \
                             const size_t *shapeA, const size_t *stridesA, size_t nDimsA, \
                             const size_t *shapeB, const size_t *stridesB, size_t nDimsB, \
                             const size_t *shapeC, const size_t *stridesC, size_t nDimsC) { \
    size_t vecSize = OPERANDS_SIZE / sizeof(type);      \
    auto *outputIndex = new size_t[nDimsC]; \
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);    \
    if (nDimsA <= 1 || nDimsB <= 1) {\
        return false; /* tblas_##op_name##_nd_by_scalar handles this */\
    }\
    if (!unalteredStrides(stridesC, shapeC, nDimsC)) { \
        return false;\
    } \
    size_t nElementsInLastDimA = shapeA[nDimsA - 1]; \
    size_t nElementsInLastDimB = shapeB[nDimsB - 1]; \
    if (stridesA[nDimsA - 1] != 1) { \
        return false; \
    } \
    if (stridesB[nDimsB - 1] != 1) { \
        return false; \
    } \
    if (!(nElementsInLastDimA == nElementsInLastDimB || nElementsInLastDimB == 1 || nElementsInLastDimA == 1)) { \
        return false; \
    } \
    size_t nElementsInLastDim = std::max(nElementsInLastDimA, nElementsInLastDimB); \
 \
    if (nElementsInLastDim < vecSize) { \
        return false; \
    } \
    bool aIsScalarInLastDim = nElementsInLastDimA == 1; \
    bool bIsScalarInLastDim = nElementsInLastDimB == 1; \
 \
    size_t nElementsC = 1; \
    for (int i = 0; i < nDimsC; i++) { \
        nElementsC *= shapeC[i]; \
    } \
    size_t cFlatIndex = 0; \
    size_t nChunks = nElementsInLastDim / vecSize; \
    size_t nRemainder = nElementsInLastDim % vecSize; \
 \
    if (!aIsScalarInLastDim && !bIsScalarInLastDim) { \
        do { \
            size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC); \
            size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC); \
\
            for (int i = 0; i < nChunks; i++) {\
                float32x4_t aVec = vld1q_f32(&a[aIndexFlat + i * vecSize]);\
                float32x4_t bVec = vld1q_f32(&b[bIndexFlat + i * vecSize]);\
                float32x4_t cVec = vec_inst(aVec, bVec);\
                vst1q_f32(&c[cFlatIndex + i * vecSize], cVec);\
            }\
            \
            for (int i = 0; i < nRemainder; i++) { \
                c[cFlatIndex + nChunks * vecSize + i] = a[aIndexFlat + nChunks * vecSize + i] scalar_op b[bIndexFlat + nChunks * vecSize + i]; \
            } \
            cFlatIndex += nElementsInLastDim; \
            outputIndex[nDimsC - 1] = nElementsInLastDim - 1; \
            incrementIndex(outputIndex, shapeC, nDimsC); \
        } while (cFlatIndex < nElementsC); \
    } else if (!aIsScalarInLastDim) { \
        do { \
            size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC); \
            size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC); \
 \
            float32x4_t bScalar = vdupq_n_f32(b[bIndexFlat]); \
            for (int i = 0; i < nChunks; i++) { \
                float32x4_t aVec = vld1q_f32(&a[aIndexFlat + i * vecSize]); \
                float32x4_t cVec = vec_inst(aVec, bScalar); \
                vst1q_f32(&c[cFlatIndex + i * vecSize], cVec); \
            } \
 \
            for (int i = 0; i < nRemainder; i++) { \
                c[cFlatIndex + nChunks * vecSize + i] = a[aIndexFlat + nChunks * vecSize + i] scalar_op b[bIndexFlat]; \
            } \
            cFlatIndex += nElementsInLastDim; \
            outputIndex[nDimsC - 1] = nElementsInLastDim - 1; \
            incrementIndex(outputIndex, shapeC, nDimsC); \
        } while (cFlatIndex < nElementsC); \
    } else if (!bIsScalarInLastDim) { \
        do { \
            size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC); \
            size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC); \
            float32x4_t aScalar = vdupq_n_f32(a[aIndexFlat]); \
            for (int i = 0; i < nChunks; i++) { \
                float32x4_t bVec = vld1q_f32(&b[bIndexFlat + i * vecSize]); \
                float32x4_t cVec = vec_inst(aScalar, bVec); \
                vst1q_f32(&c[cFlatIndex + i * vecSize], cVec); \
            }\
            for (int i = 0; i < nRemainder; i++) { \
                c[cFlatIndex + nChunks * vecSize + i] = a[aIndexFlat] scalar_op b[bIndexFlat + nChunks * vecSize + i]; \
            } \
            cFlatIndex += nElementsInLastDim; \
            outputIndex[nDimsC - 1] = nElementsInLastDim - 1; \
            incrementIndex(outputIndex, shapeC, nDimsC); \
        } while (cFlatIndex < nElementsC); \
    } else { \
        return false; \
    } \
    return true; \
}

#define op_hook_optimizations(op_name, type, optimizations, fallback_impl)\
template<>\
void tblas_##op_name(const type *a, const type *b, type *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
\
    /* check for possible optimizations */ \
    optimizations\
    fallback_impl\
}
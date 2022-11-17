#pragma once

#include <cstdint>
#include <cstdlib>

#define BINARY_OPERATION_FOR_ALL_DATA_TYPES_PROTO(operation_name) \
void operation_name(const int8_t *a, const int8_t *b, int8_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int8_t *a, const int16_t *b, int16_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int8_t *a, const int32_t *b, int32_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int8_t *a, const int64_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int8_t *a, const float *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int8_t *a, const double *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const int8_t *b, int16_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const int16_t *b, int16_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const int32_t *b, int32_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const int64_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const float *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const double *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const int8_t *b, int32_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const int16_t *b, int32_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const int32_t *b, int32_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const int64_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const float *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const double *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const int8_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const int16_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const int32_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const int64_t *b, int64_t *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const float *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const double *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const int8_t *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const int16_t *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const int32_t *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const int64_t *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const float *b, float *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const double *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const double *a, const int8_t *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const double *a, const int16_t *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const double *a, const int32_t *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const double *a, const int64_t *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\
void operation_name(const double *a, const float *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);           \
void operation_name(const double *a, const double *b, double *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC);\


#define BINARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(operation_name) \
void operation_name(const int8_t *a, const int8_t *b, int8_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, int8_t, int8_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int8_t *a, const int16_t *b, int16_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, int16_t, int16_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int8_t *a, const int32_t *b, int32_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, int32_t, int32_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int8_t *a, const int64_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, int64_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int8_t *a, const float *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, float, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int8_t *a, const double *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const int8_t *b, int16_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, int8_t, int16_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const int16_t *b, int16_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, int16_t, int16_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const int32_t *b, int32_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, int32_t, int32_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const int64_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, int64_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const float *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, float, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const double *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int32_t *a, const int8_t *b, int32_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, int8_t, int32_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int32_t *a, const int16_t *b, int32_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, int16_t, int32_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int32_t *a, const int32_t *b, int32_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, int32_t, int32_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int32_t *a, const int64_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, int64_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int32_t *a, const float *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, float, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int32_t *a, const double *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int64_t *a, const int8_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, int8_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int64_t *a, const int16_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, int16_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int64_t *a, const int32_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, int32_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int64_t *a, const int64_t *b, int64_t *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, int64_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int64_t *a, const float *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, float, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int64_t *a, const double *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const float *a, const int8_t *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<float, int8_t, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const float *a, const int16_t *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<float, int16_t, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const float *a, const int32_t *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<float, int32_t, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const float *a, const int64_t *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<float, int64_t, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const float *a, const float *b, float *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<float, float, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const float *a, const double *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<float, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const int8_t *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<double, int8_t, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const int16_t *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<double, int16_t, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const int32_t *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<double, int32_t, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const int64_t *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<double, int64_t, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const float *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<double, float, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const double *b, double *c, const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB, const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
    operation_name<double, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\

#define UNARY_OPERATION_FOR_ALL_DATA_TYPES_PROTO(operation_name) \
void operation_name(const int8_t *a, int8_t *c, size_t nElements); \
void operation_name(const int16_t *a, int16_t *c, size_t nElements); \
void operation_name(const int32_t *a, int32_t *c, size_t nElements); \
void operation_name(const int64_t *a, int64_t *c, size_t nElements); \
void operation_name(const float *a, float *c, size_t nElements); \
void operation_name(const double *a, double *c, size_t nElements); \

#define UNARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(operation_name) \
void operation_name(const int8_t *a, int8_t *c, size_t nElements) {\
    operation_name<int8_t>(a, c, nElements);\
} \
void operation_name(const int16_t *a, int16_t *c, size_t nElements) {\
    operation_name<int16_t>(a, c, nElements);\
} \
void operation_name(const int32_t *a, int32_t *c, size_t nElements) {\
    operation_name<int32_t>(a, c, nElements);\
} \
void operation_name(const int64_t *a, int64_t *c, size_t nElements) {\
    operation_name<int64_t>(a, c, nElements);\
} \
void operation_name(const float *a, float *c, size_t nElements) {\
    operation_name<float>(a, c, nElements);\
} \
void operation_name(const double *a, double *c, size_t nElements) {\
    operation_name<double>(a, c, nElements);\
}

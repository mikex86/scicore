#pragma once

#include <cstdint>

#define OPERATION_FOR_ALL_DATA_TYPES_PROTO(operation_name) \
void operation_name(const int8_t *a, const int8_t *b, int8_t *c,\
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,\
                    size_t *shapeC, size_t *stridesC, size_t nDimsC);\
void operation_name(const int16_t *a, const int16_t *b, int16_t *c,\
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,\
                    size_t *shapeC, size_t *stridesC, size_t nDimsC);\
void operation_name(const int32_t *a, const int32_t *b, int32_t *c,\
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,\
                    size_t *shapeC, size_t *stridesC, size_t nDimsC);\
void operation_name(const int64_t *a, const int64_t *b, int64_t *c,\
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,\
                    size_t *shapeC, size_t *stridesC, size_t nDimsC);\
void operation_name(const float *a, const float *b, float *c,\
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,\
                    size_t *shapeC, size_t *stridesC, size_t nDimsC);\
void operation_name(const double *a, const double *b, double *c,\
                    size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB,\
                    size_t *shapeC, size_t *stridesC, size_t nDimsC);\

#define OPERATION_FOR_ALL_DATA_TYPES_IMPL(operation_name) \
void operation_name(const int8_t *a, const int8_t *b, int8_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB, size_t *shapeC, size_t *stridesC, size_t nDimsC) {\
    operation_name<int8_t, int8_t, int8_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\
void operation_name(const int16_t *a, const int16_t *b, int16_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB, size_t *shapeC, size_t *stridesC, size_t nDimsC) {\
    operation_name<int16_t, int16_t, int16_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC, nDimsC);\
}\
void operation_name(const int32_t *a, const int32_t *b, int32_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB, size_t *shapeC, size_t *stridesC, size_t nDimsC) {\
    operation_name<int32_t, int32_t, int32_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC, nDimsC);\
}\
void operation_name(const int64_t *a, const int64_t *b, int64_t *c, size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB, size_t *shapeC, size_t *stridesC, size_t nDimsC) {\
    operation_name<int64_t, int64_t, int64_t>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC, nDimsC);\
}\
void operation_name(const float *a, const float *b, float *c, size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB, size_t *shapeC, size_t *stridesC, size_t nDimsC) {\
    operation_name<float, float, float>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC,\
            nDimsC);\
}\
void operation_name(const double *a, const double *b, double *c, size_t *shapeA, size_t *stridesA, size_t nDimsA,\
                    size_t *shapeB, size_t *stridesB, size_t nDimsB, size_t *shapeC, size_t *stridesC, size_t nDimsC) {\
    operation_name<double, double, double>(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC,\
            stridesC,\
            nDimsC);\
}\

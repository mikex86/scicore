#pragma once

#include <cstdlib>

void tblas_reducesum(const int8_t *a, int8_t *c,
                     size_t *shapeA, size_t *stridesA, size_t nDimsA,
                     size_t *shapeC, size_t *stridesC, size_t nDimsC,
                     int64_t dimension, bool keepDims);

void tblas_reducesum(const int16_t *a, int16_t *c,
                        size_t *shapeA, size_t *stridesA, size_t nDimsA,
                        size_t *shapeC, size_t *stridesC, size_t nDimsC,
                        int64_t dimension, bool keepDims);

void tblas_reducesum(const int32_t *a, int32_t *c,
                        size_t *shapeA, size_t *stridesA, size_t nDimsA,
                        size_t *shapeC, size_t *stridesC, size_t nDimsC,
                        int64_t dimension, bool keepDims);

void tblas_reducesum(const int64_t *a, int64_t *c,
                        size_t *shapeA, size_t *stridesA, size_t nDimsA,
                        size_t *shapeC, size_t *stridesC, size_t nDimsC,
                        int64_t dimension, bool keepDims);

void tblas_reducesum(const float *a, float *c,
                        size_t *shapeA, size_t *stridesA, size_t nDimsA,
                        size_t *shapeC, size_t *stridesC, size_t nDimsC,
                        int64_t dimension, bool keepDims);

void tblas_reducesum(const double *a, double *c,
                        size_t *shapeA, size_t *stridesA, size_t nDimsA,
                        size_t *shapeC, size_t *stridesC, size_t nDimsC,
                        int64_t dimension, bool keepDims);
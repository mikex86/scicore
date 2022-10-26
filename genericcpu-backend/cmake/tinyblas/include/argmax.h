#pragma once

#include <cstdlib>
#include <cstdint>

void tblas_argmax(const int8_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension);

void tblas_argmax(const int16_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension);

void tblas_argmax(const int32_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension);

void tblas_argmax(const int64_t *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension);

void tblas_argmax(const float *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension);

void tblas_argmax(const double *a, uint64_t *c,
                  size_t *shapeA, size_t *stridesA, size_t nDimsA,
                  size_t *shapeC, size_t *stridesC, size_t nDimsC,
                  int64_t dimension);
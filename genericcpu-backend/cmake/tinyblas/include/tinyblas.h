#pragma once

#include <cstdint>

enum TblasTranspose {
    TblasNoTrans = 111,
    TblasTrans = 112
};

enum TblasOrder {
    TblasRowMajor = 101,
    TblasColMajor = 102
};


// Standard BLAS functions

void tblas_sgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                 int m, int n, int k,
                 float alpha, const float *a, int lda,
                 const float *b, int ldb, float beta,
                 float *c, int ldc);

void tblas_dgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                 int m, int n, int k,
                 double alpha, double *a, int lda,
                 double *b, int ldb, double beta,
                 double *c, int ldc);

// Extended BLAS functions
void tblas_bbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, int8_t *a, int lda,
                  int8_t *b, int ldb, int8_t beta,
                  int8_t *c, int ldc);

void tblas_bsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, int8_t *a, int lda,
                  int16_t *b, int ldb, int16_t beta,
                  int16_t *c, int ldc);

void tblas_bigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, int8_t *a, int lda,
                  int32_t *b, int ldb, int32_t beta,
                  int32_t *c, int ldc);

void tblas_blgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, int8_t *a, int lda,
                  int64_t *b, int ldb, int64_t beta,
                  int64_t *c, int ldc);

void tblas_sbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int16_t alpha, int16_t *a, int lda,
                  int8_t *b, int ldb, int8_t beta,
                  int16_t *c, int ldc);

void tblas_ssgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int16_t alpha, int16_t *a, int lda,
                  int16_t *b, int ldb, int16_t beta,
                  int16_t *c, int ldc);

void tblas_sigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int16_t alpha, int16_t *a, int lda,
                  int32_t *b, int ldb, int32_t beta,
                  int32_t *c, int ldc);

void tblas_slgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int16_t alpha, int16_t *a, int lda,
                  int64_t *b, int ldb, int64_t beta,
                  int64_t *c, int ldc);

void tblas_ibgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int32_t alpha, int32_t *a, int lda,
                  int8_t *b, int ldb, int8_t beta,
                  int32_t *c, int ldc);

void tblas_isgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int32_t alpha, int32_t *a, int lda,
                  int16_t *b, int ldb, int16_t beta,
                  int32_t *c, int ldc);

void tblas_iigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int32_t alpha, int32_t *a, int lda,
                  int32_t *b, int ldb, int32_t beta,
                  int32_t *c, int ldc);

void tblas_ilgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int32_t alpha, int32_t *a, int lda,
                  int64_t *b, int ldb, int64_t beta,
                  int64_t *c, int ldc);

void tblas_lbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int64_t alpha, int64_t *a, int lda,
                  int8_t *b, int ldb, int8_t beta,
                  int64_t *c, int ldc);

void tblas_lsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int64_t alpha, int64_t *a, int lda,
                  int16_t *b, int ldb, int16_t beta,
                  int64_t *c, int ldc);

void tblas_ligemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int64_t alpha, int64_t *a, int lda,
                  int32_t *b, int ldb, int32_t beta,
                  int64_t *c, int ldc);

void tblas_llgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int64_t alpha, int64_t *a, int lda,
                  int64_t *b, int ldb, int64_t beta,
                  int64_t *c, int ldc);
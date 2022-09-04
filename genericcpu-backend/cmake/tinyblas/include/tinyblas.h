#pragma once

#include <cstdint>

enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112
};

enum CblasOrder {
    CblasRowMajor = 101,
    CblasColMajor = 102
};


void cblas_sgemm(CblasOrder order, CblasTranspose transa, CblasTranspose transb,
                 uint64_t m, uint64_t n, uint64_t k,
                 float alpha, const float *a, uint64_t lda,
                 const float *b, uint64_t ldb, float beta,
                 float *c, uint64_t ldc);

void cblas_dgemm(CblasOrder order, CblasTranspose transa, CblasTranspose transb,
                 uint64_t m, uint64_t n, uint64_t k,
                 double alpha, double *a, uint64_t lda,
                 double *b, uint64_t ldb, double beta,
                 double *c, uint64_t ldc);
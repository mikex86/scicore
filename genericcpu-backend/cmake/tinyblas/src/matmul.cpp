#include <cstdint>
#include "tinyblas.h"

void cblas_sgemm(CblasOrder order, CblasTranspose transa, CblasTranspose transb,
                 uint64_t m, uint64_t n, uint64_t k,
                 float alpha, const float *a, uint64_t lda,
                 const float *b, uint64_t ldb, float beta,
                 float *c, uint64_t ldc) {
    // TODO: WHAT TO DO WITH lda, ldb, ldc?
    for (int row = 0; row < m; row++) {
        for (int inner = 0; inner < k; inner++) {
            for (int column = 0; column < n; column++) {
                c[row * m + column] = c[row * m + column] * beta + alpha * a[row * k + inner] * b[inner * n + column];
            }
        }
    }
}

void cblas_dgemm(CblasOrder order, CblasTranspose transa, CblasTranspose transb,
                 uint64_t m, uint64_t n, uint64_t k,
                 double alpha, double *a, uint64_t lda,
                 double *b, uint64_t ldb, double beta,
                 double *c, uint64_t ldc) {
    for (int row = 0; row < m; row++) {
        for (int inner = 0; inner < k; inner++) {
            for (int column = 0; column < n; column++) {
                c[row * m + column] = c[row * m + column] * beta + alpha * a[row * k + inner] * b[inner * n + column];
            }
        }
    }
}


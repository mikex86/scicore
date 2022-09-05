#include <algorithm>
#include "tinyblas.h"

const int TILE_SIZE_FLOAT = 24; // Tested to be optimal on M1 Pro
const int TILE_SIZE_DOUBLE = 12; // Tested to be optimal on M1 Pro

// TODO: SUPPORT TRANSPOSED MATRICES

// Cache-aware matrix multiplication, based on
// "Fast Matrix Multiplication on CPU from Scratch"
// by Siboehm, https://siboehm.com/articles/22/Fast-MMM-on-CPU
// Employs:
// - Loop reordering (RIC) + L1 tiling on I
void tblas_sgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                 int m, int n, int k,
                 float alpha, const float *a, int lda,
                 const float *b, int ldb, float beta,
                 float *c, int ldc) {
    // TODO: WHAT TO DO WITH lda, ldb, ldc?
    for (int innerTile = 0; innerTile < k; innerTile += TILE_SIZE_FLOAT) {
        for (int row = 0; row < m; row++) {
            int innerTileEnd = std::min(k, innerTile + TILE_SIZE_FLOAT);
            for (int inner = innerTile; inner < innerTileEnd; inner++) {
                for (int column = 0; column < n; column++) {
                    c[row * m + column] =
                            c[row * m + column] * beta + alpha * a[row * k + inner] * b[inner * n + column];
                }
            }
        }
    }
}

void tblas_dgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                 int m, int n, int k,
                 double alpha, double *a, int lda,
                 double *b, int ldb, double beta,
                 double *c, int ldc) {
    for (int innerTile = 0; innerTile < k; innerTile += TILE_SIZE_DOUBLE) {
        for (int row = 0; row < m; row++) {
            int innerTileEnd = std::min(k, innerTile + TILE_SIZE_DOUBLE);
            for (int inner = innerTile; inner < innerTileEnd; inner++) {
                for (int column = 0; column < n; column++) {
                    c[row * m + column] =
                            c[row * m + column] * beta + alpha * a[row * k + inner] * b[inner * n + column];
                }
            }
        }
    }
}

// Extended BLAS functions
template<typename A, typename B, typename C>
inline __attribute__((always_inline)) void tblas_gemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                                                      int m, int n, int k,
                                                      A alpha, A *a, int lda,
                                                      B *b, int ldb, B beta,
                                                      C *c, int ldc) {
    // Horrible way of computing semi-optimal TILE_SIZE derived from sizeof of the respective types A, B and C
    constexpr int TILE_SIZE = (sizeof(A) + sizeof(B) + sizeof(C)) * 2;
    for (int innerTile = 0; innerTile < k; innerTile += TILE_SIZE) {
        for (int row = 0; row < m; row++) {
            int innerTileEnd = std::min(k, innerTile + TILE_SIZE);
            for (int inner = innerTile; inner < innerTileEnd; inner++) {
                for (int column = 0; column < n; column++) {
                    c[row * m + column] =
                            c[row * m + column] * beta + alpha * a[row * k + inner] * b[inner * n + column];
                }
            }
        }
    }
}

void tblas_bbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, int8_t *a, int lda,
                  int8_t *b, int ldb, int8_t beta,
                  int8_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, int8_t *a, int lda,
                  int16_t *b, int ldb, int16_t beta,
                  int16_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  int8_t *a, int lda, int32_t *b, int ldb, int32_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_blgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  int8_t *a, int lda, int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  int16_t *a, int lda, int8_t *b, int ldb, int8_t beta, int16_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ssgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  int16_t *a, int lda, int16_t *b, int ldb, int16_t beta, int16_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  int16_t *a, int lda, int32_t *b, int ldb, int32_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_slgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  int16_t *a, int lda, int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ibgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  int32_t *a, int lda, int8_t *b, int ldb, int8_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_isgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  int32_t *a, int lda, int16_t *b, int ldb, int16_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_iigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  int32_t *a, int lda, int32_t *b, int ldb, int32_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ilgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  int32_t *a, int lda, int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_lbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  int64_t *a, int lda, int8_t *b, int ldb, int8_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_lsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  int64_t *a, int lda, int16_t *b, int ldb, int16_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ligemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  int64_t *a, int lda, int32_t *b, int ldb, int32_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_llgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  int64_t *a, int lda, int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

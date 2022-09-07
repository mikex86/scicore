#include <algorithm>
#include "tinyblas.h"

#define FORCE_INLINE inline

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
FORCE_INLINE void tblas_gemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                             int m, int n, int k,
                             A alpha, A *a, int lda,
                             B *b, int ldb, B beta,
                             C *c, int ldc) {
    constexpr int TILE_SIZE = 16;
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

void tblas_bfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  int8_t *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bdgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  int8_t *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
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

void tblas_sfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  int16_t *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sdgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  int16_t *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
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

void tblas_ifgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  int32_t *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_idgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  int32_t *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
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

void tblas_lfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  int64_t *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ldgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  int64_t *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


void
tblas_fbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha, float *a,
             int lda, int8_t *b, int ldb, int8_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_fsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha, float *a,
             int lda, int16_t *b, int ldb, int16_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_figemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha, float *a,
             int lda, int32_t *b, int ldb, int32_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_flgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha, float *a,
             int lda, int64_t *b, int ldb, int64_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_fdgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha, float *a,
             int lda, double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  double *a, int lda, int8_t *b, int ldb, int8_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  double *a, int lda, int16_t *b, int ldb, int16_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_digemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  double *a, int lda, int32_t *b, int ldb, int32_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dlgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  double *a, int lda, int64_t *b, int ldb, int64_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  double *a, int lda, float *b, int ldb, float beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

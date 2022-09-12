#include <algorithm>
#include "tinyblas.h"

#define FORCE_INLINE inline

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_gemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                             int m, int n, int k,
                             A alpha, const A *a, int lda,
                             const B *b, int ldb, B beta,
                             C *c, int ldc) {
    if (beta != 1.0f) {
        if (beta == 0.0f) {
            memset(c, 0, m * n * sizeof(float));
        } else {
            for (int i = 0; i < m * n; i++) {
                c[i] *= beta;
            }
        }
    }
    switch (order) {
        case TblasRowMajor: {
            for (int row = 0; row < m; row++) {
                for (int inner = 0; inner < k; inner++) {
                    for (int col = 0; col < n; col++) {
                        int aIdx = transa == TblasNoTrans ? row * lda + inner : inner * lda + row;
                        int bIdx = transb == TblasNoTrans ? inner * ldb + col : col * ldb + inner;
                        c[row * ldc + col] += alpha * a[aIdx] * beta * b[bIdx];
                    }
                }
            }
            break;
        }
        case TblasColMajor: {
            for (int row = 0; row < m; row++) {
                for (int inner = 0; inner < k; inner++) {
                    for (int col = 0; col < n; col++) {
                        int aIdx = transa == TblasNoTrans ? inner * lda + row : row * lda + inner;
                        int bIdx = transb == TblasNoTrans ? inner * ldb + col : col * ldb + inner;
                        c[row * ldc + col] += alpha * a[aIdx] * b[bIdx];
                    }
                }
            }
            break;
        }
    }
}

void tblas_sgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                 int m, int n, int k,
                 float alpha, const float *a, int lda,
                 const float *b, int ldb, float beta,
                 float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dgemm(TblasOrder order, TblasTranspose transpose, TblasTranspose transb,
                 int m, int n, int k,
                 double alpha, const double *a, int lda,
                 const double *b, int ldb, double beta,
                 double *c, int ldc) {
    tblas_gemm(order, transpose, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// Extended BLAS functions
void tblas_bbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, const int8_t *a, int lda,
                  const int8_t *b, int ldb, int8_t beta,
                  int8_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb,
                  int m, int n, int k,
                  int8_t alpha, const int8_t *a, int lda,
                  const int16_t *b, int ldb, int16_t beta,
                  int16_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  const int8_t *a, int lda, const int32_t *b, int ldb, int32_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_blgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  const int8_t *a, int lda, const int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  const int8_t *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_bdgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int8_t alpha,
                  const int8_t *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  const int16_t *a, int lda, const int8_t *b, int ldb, int8_t beta, int16_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ssgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  const int16_t *a, int lda, const int16_t *b, int ldb, int16_t beta, int16_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  const int16_t *a, int lda, const int32_t *b, int ldb, int32_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_slgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  const int16_t *a, int lda, const int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  const int16_t *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_sdgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int16_t alpha,
                  const int16_t *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ibgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  const int32_t *a, int lda, const int8_t *b, int ldb, int8_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_isgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  const int32_t *a, int lda, const int16_t *b, int ldb, int16_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_iigemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  const int32_t *a, int lda, const int32_t *b, int ldb, int32_t beta, int32_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ilgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  const int32_t *a, int lda, const int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ifgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  const int32_t *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_idgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int32_t alpha,
                  const int32_t *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_lbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  const int64_t *a, int lda, const int8_t *b, int ldb, int8_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_lsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  const int64_t *a, int lda, const int16_t *b, int ldb, int16_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ligemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  const int64_t *a, int lda, const int32_t *b, int ldb, int32_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_llgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  const int64_t *a, int lda, const int64_t *b, int ldb, int64_t beta, int64_t *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_lfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  const int64_t *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_ldgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, int64_t alpha,
                  const int64_t *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


void
tblas_fbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha,
             const float *a, int lda, const int8_t *b, int ldb, int8_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_fsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha,
             const float *a,
             int lda, const int16_t *b, int ldb, int16_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_figemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha,
             const float *a,
             int lda, const int32_t *b, int ldb, int32_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_flgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha,
             const float *a,
             int lda, const int64_t *b, int ldb, int64_t beta, float *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tblas_fdgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, float alpha,
             const float *a,
             int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dbgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  const double *a, int lda, const int8_t *b, int ldb, int8_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dsgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  const double *a, int lda, const int16_t *b, int ldb, int16_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_digemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  const double *a, int lda, const int32_t *b, int ldb, int32_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dlgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  const double *a, int lda, const int64_t *b, int ldb, int64_t beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void tblas_dfgemm(TblasOrder order, TblasTranspose transa, TblasTranspose transb, int m, int n, int k, double alpha,
                  const double *a, int lda, const float *b, int ldb, float beta, double *c, int ldc) {
    tblas_gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#include "MatmulJNI.h"

#ifdef __APPLE__
// if macOS, use Accelerate framework
#include <Accelerate/Accelerate.h>
#else
#define USE_TINYBLAS
#endif

#include <matmul.h>

#define OP_NONE 0
#define OP_TRANSPOSE 1

#define MATMUL_DATA_TYPE_INT8 1
#define MATMUL_DATA_TYPE_INT16 2
#define MATMUL_DATA_TYPE_INT32 3
#define MATMUL_DATA_TYPE_INT64 4
#define MATMUL_DATA_TYPE_FLOAT32 5
#define MATMUL_DATA_TYPE_FLOAT64 6

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MatmulJNI_nmatmul(JNIEnv *jniEnv, jclass, jint transa, jint transb,
                                                                     jint m, jint n, jint k,
                                                                     jlong alphaPtr,
                                                                     jlong aPtr,
                                                                     jint aType,
                                                                     jint lda,
                                                                     jlong betaPtr, jlong bPtr,
                                                                     jint bType,
                                                                     jint ldb,
                                                                     jlong cPtr,
                                                                     jint cType,
                                                                     jint ldc) {
    // TODO: MAKE matmul RESPECT STRIDES
    if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_FLOAT32 && cType == MATMUL_DATA_TYPE_FLOAT32) {
#ifdef USE_TINYBLAS
        tblas_sgemm(TblasRowMajor,
                    transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (float *) bPtr, ldb, *(float *) betaPtr,
                    (float *) cPtr, ldc);
#else
        cblas_sgemm(CblasRowMajor,
                    transa == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    transb == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (float *) bPtr, ldb, *(float *) betaPtr,
                    (float *) cPtr, ldc);
#endif
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_FLOAT64 && cType == MATMUL_DATA_TYPE_FLOAT64) {
#ifdef USE_TINYBLAS
        tblas_dgemm(TblasRowMajor,
                    transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    m, n, k,
                    *(double *) alphaPtr, (double *) aPtr, lda,
                    (double *) bPtr, ldb, *(double *) betaPtr,
                    (double *) cPtr, ldc);
#else
        cblas_dgemm(CblasRowMajor,
                    transa == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    transb == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    *(double *) alphaPtr, (double *) aPtr, lda,
                    (double *) bPtr, ldb, *(double *) betaPtr,
                    (double *) cPtr, ldc);
#endif
    }
        // Data type combinations unsupported by BLAS, only implemented in TinyBLAS
    else if (aType == MATMUL_DATA_TYPE_INT8 && bType == MATMUL_DATA_TYPE_INT8) {
        tblas_bbgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int8_t *) alphaPtr, (int8_t *) aPtr, lda,
                     (int8_t *) bPtr, ldb, *(int8_t *) betaPtr,
                     (int8_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT8 && bType == MATMUL_DATA_TYPE_INT16) {
        tblas_bsgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int8_t *) alphaPtr, (int8_t *) aPtr, lda,
                     (int16_t *) bPtr, ldb, *(int16_t *) betaPtr,
                     (int16_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT8 && bType == MATMUL_DATA_TYPE_INT32) {
        tblas_bigemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int8_t *) alphaPtr, (int8_t *) aPtr, lda,
                     (int32_t *) bPtr, ldb, *(int32_t *) betaPtr,
                     (int32_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT8 && bType == MATMUL_DATA_TYPE_INT64) {
        tblas_blgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int8_t *) alphaPtr, (int8_t *) aPtr, lda,
                     (int64_t *) bPtr, ldb, *(int64_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT8 && bType == MATMUL_DATA_TYPE_FLOAT32) {
        tblas_bfgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int8_t *) alphaPtr, (int8_t *) aPtr, lda,
                     (float *) bPtr, ldb, *(float *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT8 && bType == MATMUL_DATA_TYPE_FLOAT64) {
        tblas_bdgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int8_t *) alphaPtr, (int8_t *) aPtr, lda,
                     (double *) bPtr, ldb, *(double *) betaPtr,
                     (double *) cPtr, ldc);
    }
    else if (aType == MATMUL_DATA_TYPE_INT16 && bType == MATMUL_DATA_TYPE_INT8) {
        tblas_sbgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int16_t *) alphaPtr, (int16_t *) aPtr, lda,
                     (int8_t *) bPtr, ldb, *(int8_t *) betaPtr,
                     (int16_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT16 && bType == MATMUL_DATA_TYPE_INT16) {
        tblas_ssgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int16_t *) alphaPtr, (int16_t *) aPtr, lda,
                     (int16_t *) bPtr, ldb, *(int16_t *) betaPtr,
                     (int16_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT16 && bType == MATMUL_DATA_TYPE_INT32) {
        tblas_sigemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int16_t *) alphaPtr, (int16_t *) aPtr, lda,
                     (int32_t *) bPtr, ldb, *(int32_t *) betaPtr,
                     (int32_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT16 && bType == MATMUL_DATA_TYPE_INT64) {
        tblas_slgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int16_t *) alphaPtr, (int16_t *) aPtr, lda,
                     (int64_t *) bPtr, ldb, *(int64_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT16 && bType == MATMUL_DATA_TYPE_FLOAT32) {
        tblas_sfgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int16_t *) alphaPtr, (int16_t *) aPtr, lda,
                     (float *) bPtr, ldb, *(float *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT16 && bType == MATMUL_DATA_TYPE_FLOAT64) {
        tblas_sdgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int16_t *) alphaPtr, (int16_t *) aPtr, lda,
                     (double *) bPtr, ldb, *(double *) betaPtr,
                     (double *) cPtr, ldc);
    }
    else if (aType == MATMUL_DATA_TYPE_INT32 && bType == MATMUL_DATA_TYPE_INT8) {
        tblas_ibgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int32_t *) alphaPtr, (int32_t *) aPtr, lda,
                     (int8_t *) bPtr, ldb, *(int8_t *) betaPtr,
                     (int32_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT32 && bType == MATMUL_DATA_TYPE_INT16) {
        tblas_isgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int32_t *) alphaPtr, (int32_t *) aPtr, lda,
                     (int16_t *) bPtr, ldb, *(int16_t *) betaPtr,
                     (int32_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT32 && bType == MATMUL_DATA_TYPE_INT32) {
        tblas_iigemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int32_t *) alphaPtr, (int32_t *) aPtr, lda,
                     (int32_t *) bPtr, ldb, *(int32_t *) betaPtr,
                     (int32_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT32 && bType == MATMUL_DATA_TYPE_INT64) {
        tblas_ilgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int32_t *) alphaPtr, (int32_t *) aPtr, lda,
                     (int64_t *) bPtr, ldb, *(int64_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT32 && bType == MATMUL_DATA_TYPE_FLOAT32) {
        tblas_ifgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int32_t *) alphaPtr, (int32_t *) aPtr, lda,
                     (float *) bPtr, ldb, *(float *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT32 && bType == MATMUL_DATA_TYPE_FLOAT64) {
        tblas_idgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int32_t *) alphaPtr, (int32_t *) aPtr, lda,
                     (double *) bPtr, ldb, *(double *) betaPtr,
                     (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT64 && bType == MATMUL_DATA_TYPE_INT8) {
        tblas_lbgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int64_t *) alphaPtr, (int64_t *) aPtr, lda,
                     (int8_t *) bPtr, ldb, *(int8_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT64 && bType == MATMUL_DATA_TYPE_INT16) {
        tblas_lsgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int64_t *) alphaPtr, (int64_t *) aPtr, lda,
                     (int16_t *) bPtr, ldb, *(int16_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT64 && bType == MATMUL_DATA_TYPE_INT32) {
        tblas_ligemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int64_t *) alphaPtr, (int64_t *) aPtr, lda,
                     (int32_t *) bPtr, ldb, *(int32_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT64 && bType == MATMUL_DATA_TYPE_INT64) {
        tblas_llgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int64_t *) alphaPtr, (int64_t *) aPtr, lda,
                     (int64_t *) bPtr, ldb, *(int64_t *) betaPtr,
                     (int64_t *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT64 && bType == MATMUL_DATA_TYPE_FLOAT32) {
        tblas_lfgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int64_t *) alphaPtr, (int64_t *) aPtr, lda,
                     (float *) bPtr, ldb, *(float *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_INT64 && bType == MATMUL_DATA_TYPE_FLOAT64) {
        tblas_ldgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(int64_t *) alphaPtr, (int64_t *) aPtr, lda,
                     (double *) bPtr, ldb, *(double *) betaPtr,
                     (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_INT8) {
        tblas_fbgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(float *) alphaPtr, (float *) aPtr, lda,
                     (int8_t *) bPtr, ldb, *(int8_t *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_INT16) {
        tblas_fsgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(float *) alphaPtr, (float *) aPtr, lda,
                     (int16_t *) bPtr, ldb, *(int16_t *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_INT32) {
        tblas_figemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                        transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                        m, n, k,
                        *(float *) alphaPtr, (float *) aPtr, lda,
                        (int32_t *) bPtr, ldb, *(int32_t *) betaPtr,
                        (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_INT64) {
        tblas_flgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(float *) alphaPtr, (float *) aPtr, lda,
                     (int64_t *) bPtr, ldb, *(int64_t *) betaPtr,
                     (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_FLOAT32) {
        tblas_sgemm(TblasRowMajor,
                    transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (float *) bPtr, ldb, *(float *) betaPtr,
                    (float *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT32 && bType == MATMUL_DATA_TYPE_FLOAT64) {
        tblas_fdgemm(TblasRowMajor,
                    transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (double *) bPtr, ldb, *(double *) betaPtr,
                    (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_INT8) {
        tblas_dbgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(double *) alphaPtr, (double *) aPtr, lda,
                     (int8_t *) bPtr, ldb, *(int8_t *) betaPtr,
                     (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_INT16) {
        tblas_dsgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                        m, n, k,
                        *(double *) alphaPtr, (double *) aPtr, lda,
                        (int16_t *) bPtr, ldb, *(int16_t *) betaPtr,
                        (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_INT32) {
        tblas_digemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(double *) alphaPtr, (double *) aPtr, lda,
                     (int32_t *) bPtr, ldb, *(int32_t *) betaPtr,
                     (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_INT64) {
        tblas_dlgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(double *) alphaPtr, (double *) aPtr, lda,
                     (int64_t *) bPtr, ldb, *(int64_t *) betaPtr,
                     (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_FLOAT32) {
        tblas_dfgemm(TblasRowMajor,
                     transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                     m, n, k,
                     *(double *) alphaPtr, (double *) aPtr, lda,
                     (float *) bPtr, ldb, *(float *) betaPtr,
                     (double *) cPtr, ldc);
    } else if (aType == MATMUL_DATA_TYPE_FLOAT64 && bType == MATMUL_DATA_TYPE_FLOAT64) {
        tblas_dgemm(TblasRowMajor,
                    transa == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    transb == OP_TRANSPOSE ? TblasTrans : TblasNoTrans,
                    m, n, k,
                    *(double *) alphaPtr, (double *) aPtr, lda,
                    (double *) bPtr, ldb, *(double *) betaPtr,
                    (double *) cPtr, ldc);
    } else {
        jniEnv->ThrowNew(jniEnv->FindClass("java/lang/UnsupportedOperationException"),
                         "Unsupported data type combination");
    }
}
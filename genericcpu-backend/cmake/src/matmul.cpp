#include "matmul.h"

#ifdef __APPLE__
// if macOS, use Accelerate framework
#include <Accelerate/Accelerate.h>
#else
#include <tinyblas.h>
#include <cstdint>

#endif

#define OP_NONE 0
#define OP_TRANSPOSE 1

#define DATA_TYPE_INT8 1
#define DATA_TYPE_INT16 2
#define DATA_TYPE_INT32 3
#define DATA_TYPE_INT64 4
#define DATA_TYPE_FLOAT32 5
#define DATA_TYPE_FLOAT64 6

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MatmulJNI_matmul(JNIEnv *, jclass, jint transa, jint transb,
                                                                     jlong m, jlong n, jlong k,
                                                                     jlong alphaPtr,
                                                                     jlong aPtr,
                                                                     jint aType,
                                                                     jlong lda,
                                                                     jlong betaPtr, jlong bPtr,
                                                                     jint bType,
                                                                     jlong ldb,
                                                                     jlong cPtr,
                                                                     jint cType,
                                                                     jlong ldc) {
    if (aType == DATA_TYPE_FLOAT32 && bType == DATA_TYPE_FLOAT32 && cType == DATA_TYPE_FLOAT32) {
        cblas_sgemm(CblasRowMajor,
                    transa == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    transb == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    *(float *) alphaPtr, (float *) aPtr, lda,
                    (float *) bPtr, ldb, *(float *) betaPtr,
                    (float *) cPtr, ldc);
    } else if (aType == DATA_TYPE_FLOAT64 && bType == DATA_TYPE_FLOAT64 && cType == DATA_TYPE_FLOAT64) {
        cblas_dgemm(CblasRowMajor,
                    transa == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    transb == OP_TRANSPOSE ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    *(double *) alphaPtr, (double *) aPtr, lda,
                    (double *) bPtr, ldb, *(double *) betaPtr,
                    (double *) cPtr, ldc);
    }
    // TODO: SUPPORT MORE DATA TYPE COMBINATIONS
}
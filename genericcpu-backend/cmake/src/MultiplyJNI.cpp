#include "MultiplyJNI.h"

#include "multiply.h"
#include <algorithm>

#define MULTIPLY_DATA_TYPE_INT8 1
#define MULTIPLY_DATA_TYPE_INT16 2
#define MULTIPLY_DATA_TYPE_INT32 3
#define MULTIPLY_DATA_TYPE_INT64 4
#define MULTIPLY_DATA_TYPE_FLOAT32 5
#define MULTIPLY_DATA_TYPE_FLOAT64 6

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MultiplyJNI_nmultiply(JNIEnv *jniEnv, jclass,
                                                                          jlong aPtr,
                                                                          jint aDataType,
                                                                          jlong nElementsA,
                                                                          jlong bPtr,
                                                                          jint bDataType,
                                                                          jlong nElementsB,
                                                                          jlong cPtr,
                                                                          jlong nElementsC) {
    // TODO: use single scalar variant when nElementsB == 1
    if (aDataType == MULTIPLY_DATA_TYPE_INT8 && bDataType == MULTIPLY_DATA_TYPE_INT8) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_bbmul((int8_t *) aPtr, *(int8_t *) bPtr, (int8_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_bbmul((int8_t *) bPtr, *(int8_t *) aPtr, (int8_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_bbmul((int8_t *) aPtr, (int8_t *) bPtr, (int8_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_bbmul_broadcast((int8_t *) aPtr, (int8_t *) bPtr, (int8_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_bbmul_broadcast((int8_t *) bPtr, (int8_t *) aPtr, (int8_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT8 && bDataType == MULTIPLY_DATA_TYPE_INT16) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_bsmul((int8_t *) aPtr, *(int16_t *) bPtr, (int16_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_sbmul((int16_t *) bPtr, *(int8_t *) aPtr, (int16_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_bsmul((int8_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_bsmul_broadcast((int8_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_sbmul_broadcast((int16_t *) bPtr, (int8_t *) aPtr, (int16_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT8 && bDataType == MULTIPLY_DATA_TYPE_INT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_bimul((int8_t *) aPtr, *(int32_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ibmul((int32_t *) bPtr, *(int8_t *) aPtr, (int32_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_bimul((int8_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_bimul_broadcast((int8_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ibmul_broadcast((int32_t *) bPtr, (int8_t *) aPtr, (int32_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT8 && bDataType == MULTIPLY_DATA_TYPE_INT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_blmul((int8_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_lbmul((int64_t *) bPtr, *(int8_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_blmul((int8_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_blmul_broadcast((int8_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_lbmul_broadcast((int64_t *) bPtr, (int8_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT8 && bDataType == MULTIPLY_DATA_TYPE_FLOAT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_bfmul((int8_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_fbmul((float *) bPtr, *(int8_t *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_bfmul((int8_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_bfmul_broadcast((int8_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_fbmul_broadcast((float *) bPtr, (int8_t *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT8 && bDataType == MULTIPLY_DATA_TYPE_FLOAT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_bdmul((int8_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_dbmul((double *) bPtr, *(int8_t *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_bdmul((int8_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_bdmul_broadcast((int8_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_dbmul_broadcast((double *) bPtr, (int8_t *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT16 && bDataType == MULTIPLY_DATA_TYPE_INT8) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_sbmul((int16_t *) aPtr, *(int8_t *) bPtr, (int16_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_bsmul((int8_t *) bPtr, *(int16_t *) aPtr, (int16_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_sbmul((int16_t *) aPtr, (int8_t *) bPtr, (int16_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_sbmul_broadcast((int16_t *) aPtr, (int8_t *) bPtr, (int16_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_bsmul_broadcast((int8_t *) bPtr, (int16_t *) aPtr, (int16_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT16 && bDataType == MULTIPLY_DATA_TYPE_INT16) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ssmul((int16_t *) aPtr, *(int16_t *) bPtr, (int16_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ssmul((int16_t *) bPtr, *(int16_t *) aPtr, (int16_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ssmul((int16_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ssmul_broadcast((int16_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ssmul_broadcast((int16_t *) bPtr, (int16_t *) aPtr, (int16_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT16 && bDataType == MULTIPLY_DATA_TYPE_INT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_simul((int16_t *) aPtr, *(int32_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ismul((int32_t *) bPtr, *(int16_t *) aPtr, (int32_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_simul((int16_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_simul_broadcast((int16_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ismul_broadcast((int32_t *) bPtr, (int16_t *) aPtr, (int32_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT16 && bDataType == MULTIPLY_DATA_TYPE_INT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_slmul((int16_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_lsmul((int64_t *) bPtr, *(int16_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_slmul((int16_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_slmul_broadcast((int16_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_lsmul_broadcast((int64_t *) bPtr, (int16_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT16 && bDataType == MULTIPLY_DATA_TYPE_FLOAT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_sfmul((int16_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_fsmul((float *) bPtr, *(int16_t *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_sfmul((int16_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_sfmul_broadcast((int16_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_fsmul_broadcast((float *) bPtr, (int16_t *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT16 && bDataType == MULTIPLY_DATA_TYPE_FLOAT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_sdmul((int16_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_dsmul((double *) bPtr, *(int16_t *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_sdmul((int16_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_sdmul_broadcast((int16_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_dsmul_broadcast((double *) bPtr, (int16_t *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT32 && bDataType == MULTIPLY_DATA_TYPE_INT8) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ibmul((int32_t *) aPtr, *(int8_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_bimul((int8_t *) bPtr, *(int32_t *) aPtr, (int32_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ibmul((int32_t *) aPtr, (int8_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ibmul_broadcast((int32_t *) aPtr, (int8_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_bimul_broadcast((int8_t *) bPtr, (int32_t *) aPtr, (int32_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT32 && bDataType == MULTIPLY_DATA_TYPE_INT16) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ismul((int32_t *) aPtr, *(int16_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_simul((int16_t *) bPtr, *(int32_t *) aPtr, (int32_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ismul((int32_t *) aPtr, (int16_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ismul_broadcast((int32_t *) aPtr, (int16_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_simul_broadcast((int16_t *) bPtr, (int32_t *) aPtr, (int32_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT32 && bDataType == MULTIPLY_DATA_TYPE_INT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_iimul((int32_t *) aPtr, *(int32_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_iimul((int32_t *) bPtr, *(int32_t *) aPtr, (int32_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_iimul((int32_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_iimul_broadcast((int32_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_iimul_broadcast((int32_t *) bPtr, (int32_t *) aPtr, (int32_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT32 && bDataType == MULTIPLY_DATA_TYPE_INT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ilmul((int32_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_limul((int64_t *) bPtr, *(int32_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ilmul((int32_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ilmul_broadcast((int32_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_limul_broadcast((int64_t *) bPtr, (int32_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT32 && bDataType == MULTIPLY_DATA_TYPE_FLOAT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ifmul((int32_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_fimul((float *) bPtr, *(int32_t *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ifmul((int32_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ifmul_broadcast((int32_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_fimul_broadcast((float *) bPtr, (int32_t *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT32 && bDataType == MULTIPLY_DATA_TYPE_FLOAT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_idmul((int32_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_dimul((double *) bPtr, *(int32_t *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_idmul((int32_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_idmul_broadcast((int32_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_dimul_broadcast((double *) bPtr, (int32_t *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT64 && bDataType == MULTIPLY_DATA_TYPE_INT8) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_lbmul((int64_t *) aPtr, *(int8_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_blmul((int8_t *) bPtr, *(int64_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_lbmul((int64_t *) aPtr, (int8_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_lbmul_broadcast((int64_t *) aPtr, (int8_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_blmul_broadcast((int8_t *) bPtr, (int64_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT64 && bDataType == MULTIPLY_DATA_TYPE_INT16) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_lsmul((int64_t *) aPtr, *(int16_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_slmul((int16_t *) bPtr, *(int64_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_lsmul((int64_t *) aPtr, (int16_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_lsmul_broadcast((int64_t *) aPtr, (int16_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_slmul_broadcast((int16_t *) bPtr, (int64_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT64 && bDataType == MULTIPLY_DATA_TYPE_INT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_limul((int64_t *) aPtr, *(int32_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ilmul((int32_t *) bPtr, *(int64_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_limul((int64_t *) aPtr, (int32_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_limul_broadcast((int64_t *) aPtr, (int32_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ilmul_broadcast((int32_t *) bPtr, (int64_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT64 && bDataType == MULTIPLY_DATA_TYPE_INT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_llmul((int64_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_llmul((int64_t *) bPtr, *(int64_t *) aPtr, (int64_t *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_llmul((int64_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_llmul_broadcast((int64_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_llmul_broadcast((int64_t *) bPtr, (int64_t *) aPtr, (int64_t *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT64 && bDataType == MULTIPLY_DATA_TYPE_FLOAT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_lfmul((int64_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_flmul((float *) bPtr, *(int64_t *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_lfmul((int64_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_lfmul_broadcast((int64_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_flmul_broadcast((float *) bPtr, (int64_t *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_INT64 && bDataType == MULTIPLY_DATA_TYPE_FLOAT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ldmul((int64_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_dlmul((double *) bPtr, *(int64_t *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ldmul((int64_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ldmul_broadcast((int64_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_dlmul_broadcast((double *) bPtr, (int64_t *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT32 && bDataType == MULTIPLY_DATA_TYPE_INT8) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_fbmul((float *) aPtr, *(int8_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_bfmul((int8_t *) bPtr, *(float *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_fbmul((float *) aPtr, (int8_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_fbmul_broadcast((float *) aPtr, (int8_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_bfmul_broadcast((int8_t *) bPtr, (float *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT32 && bDataType == MULTIPLY_DATA_TYPE_INT16) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_fsmul((float *) aPtr, *(int16_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_sfmul((int16_t *) bPtr, *(float *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_fsmul((float *) aPtr, (int16_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_fsmul_broadcast((float *) aPtr, (int16_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_sfmul_broadcast((int16_t *) bPtr, (float *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT32 && bDataType == MULTIPLY_DATA_TYPE_INT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_fimul((float *) aPtr, *(int32_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ifmul((int32_t *) bPtr, *(float *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_fimul((float *) aPtr, (int32_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_fimul_broadcast((float *) aPtr, (int32_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ifmul_broadcast((int32_t *) bPtr, (float *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT32 && bDataType == MULTIPLY_DATA_TYPE_INT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_flmul((float *) aPtr, *(int64_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_lfmul((int64_t *) bPtr, *(float *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_flmul((float *) aPtr, (int64_t *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_flmul_broadcast((float *) aPtr, (int64_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_lfmul_broadcast((int64_t *) bPtr, (float *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT32 && bDataType == MULTIPLY_DATA_TYPE_FLOAT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ffmul((float *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ffmul((float *) bPtr, *(float *) aPtr, (float *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ffmul((float *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ffmul_broadcast((float *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ffmul_broadcast((float *) bPtr, (float *) aPtr, (float *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT32 && bDataType == MULTIPLY_DATA_TYPE_FLOAT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_fdmul((float *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_dfmul((double *) bPtr, *(float *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_fdmul((float *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_fdmul_broadcast((float *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_dfmul_broadcast((double *) bPtr, (float *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT64 && bDataType == MULTIPLY_DATA_TYPE_INT8) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_dbmul((double *) aPtr, *(int8_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_bdmul((int8_t *) bPtr, *(double *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_dbmul((double *) aPtr, (int8_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_dbmul_broadcast((double *) aPtr, (int8_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_bdmul_broadcast((int8_t *) bPtr, (double *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT64 && bDataType == MULTIPLY_DATA_TYPE_INT16) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_dsmul((double *) aPtr, *(int16_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_sdmul((int16_t *) bPtr, *(double *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_dsmul((double *) aPtr, (int16_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_dsmul_broadcast((double *) aPtr, (int16_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_sdmul_broadcast((int16_t *) bPtr, (double *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT64 && bDataType == MULTIPLY_DATA_TYPE_INT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_dimul((double *) aPtr, *(int32_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_idmul((int32_t *) bPtr, *(double *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_dimul((double *) aPtr, (int32_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_dimul_broadcast((double *) aPtr, (int32_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_idmul_broadcast((int32_t *) bPtr, (double *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT64 && bDataType == MULTIPLY_DATA_TYPE_INT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_dlmul((double *) aPtr, *(int64_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ldmul((int64_t *) bPtr, *(double *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_dlmul((double *) aPtr, (int64_t *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_dlmul_broadcast((double *) aPtr, (int64_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ldmul_broadcast((int64_t *) bPtr, (double *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT64 && bDataType == MULTIPLY_DATA_TYPE_FLOAT32) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_dfmul((double *) aPtr, *(float *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_fdmul((float *) bPtr, *(double *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_dfmul((double *) aPtr, (float *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_dfmul_broadcast((double *) aPtr, (float *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_fdmul_broadcast((float *) bPtr, (double *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else if (aDataType == MULTIPLY_DATA_TYPE_FLOAT64 && bDataType == MULTIPLY_DATA_TYPE_FLOAT64) {
        if (nElementsC < std::max(nElementsA, nElementsB)) {
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                             "C array is too small");
        }
        if (nElementsB == 1) {
            tblas_tensor_ddmul((double *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA == 1) {
            tblas_tensor_ddmul((double *) bPtr, *(double *) aPtr, (double *) cPtr, nElementsB);
        } else if (nElementsA == nElementsB) {
            tblas_tensor_ddmul((double *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);
        } else if (nElementsA > nElementsB) {
            tblas_tensor_ddmul_broadcast((double *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);
        } else if (nElementsB > nElementsA) {
            tblas_tensor_ddmul_broadcast((double *) bPtr, (double *) aPtr, (double *) cPtr, nElementsB, nElementsA);
        }
    } else {
        jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),
                         "Unsupported data type");
    }
}

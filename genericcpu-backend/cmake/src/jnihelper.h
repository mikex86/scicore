#pragma once

#include <jni.h>
#include <algorithm>

#define DATA_TYPE_INT8 1
#define DATA_TYPE_INT16 2
#define DATA_TYPE_INT32 3
#define DATA_TYPE_INT64 4
#define DATA_TYPE_FLOAT32 5
#define DATA_TYPE_FLOAT64 6

#define BINARY_OP_JNI_WRAPPER_FUNC_FOR_ALL_TYPES_ALL_VARIANTS(java_func_name, op_name) \
JNIEXPORT void JNICALL java_func_name(JNIEnv *jniEnv, jclass, jlong aPtr, jint aDataType, jlong nElementsA, jlong bPtr, jint bDataType, jlong nElementsB, jlong cPtr, jlong nElementsC) { \
    if (aDataType == DATA_TYPE_INT8 && bDataType == DATA_TYPE_INT8) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_bb##op_name((int8_t *) aPtr, *(int8_t *) bPtr, (int8_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_bb##op_name(*(int8_t *) aPtr, (int8_t *) bPtr, (int8_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_bb##op_name((int8_t *) aPtr, (int8_t *) bPtr, (int8_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_bb##op_name##_broadcast_right((int8_t *) aPtr, (int8_t *) bPtr, (int8_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_bb##op_name##_broadcast_left((int8_t *) aPtr, (int8_t *) bPtr, (int8_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT8 && bDataType == DATA_TYPE_INT16) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_bs##op_name((int8_t *) aPtr, *(int16_t *) bPtr, (int16_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_bs##op_name(*(int8_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_bs##op_name((int8_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_bs##op_name##_broadcast_right((int8_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_bs##op_name##_broadcast_left((int8_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT8 && bDataType == DATA_TYPE_INT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_bi##op_name((int8_t *) aPtr, *(int32_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_bi##op_name(*(int8_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_bi##op_name((int8_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_bi##op_name##_broadcast_right((int8_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_bi##op_name##_broadcast_left((int8_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT8 && bDataType == DATA_TYPE_INT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_bl##op_name((int8_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_bl##op_name(*(int8_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_bl##op_name((int8_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_bl##op_name##_broadcast_right((int8_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_bl##op_name##_broadcast_left((int8_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT8 && bDataType == DATA_TYPE_FLOAT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_bf##op_name((int8_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_bf##op_name(*(int8_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_bf##op_name((int8_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_bf##op_name##_broadcast_right((int8_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_bf##op_name##_broadcast_left((int8_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT8 && bDataType == DATA_TYPE_FLOAT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_bd##op_name((int8_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_bd##op_name(*(int8_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_bd##op_name((int8_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_bd##op_name##_broadcast_right((int8_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_bd##op_name##_broadcast_left((int8_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT16 && bDataType == DATA_TYPE_INT8) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_sb##op_name((int16_t *) aPtr, *(int8_t *) bPtr, (int16_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_sb##op_name(*(int16_t *) aPtr, (int8_t *) bPtr, (int16_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_sb##op_name((int16_t *) aPtr, (int8_t *) bPtr, (int16_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_sb##op_name##_broadcast_right((int16_t *) aPtr, (int8_t *) bPtr, (int16_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_sb##op_name##_broadcast_left((int16_t *) aPtr, (int8_t *) bPtr, (int16_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT16 && bDataType == DATA_TYPE_INT16) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ss##op_name((int16_t *) aPtr, *(int16_t *) bPtr, (int16_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ss##op_name(*(int16_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ss##op_name((int16_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ss##op_name##_broadcast_right((int16_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ss##op_name##_broadcast_left((int16_t *) aPtr, (int16_t *) bPtr, (int16_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT16 && bDataType == DATA_TYPE_INT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_si##op_name((int16_t *) aPtr, *(int32_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_si##op_name(*(int16_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_si##op_name((int16_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_si##op_name##_broadcast_right((int16_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_si##op_name##_broadcast_left((int16_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT16 && bDataType == DATA_TYPE_INT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_sl##op_name((int16_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_sl##op_name(*(int16_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_sl##op_name((int16_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_sl##op_name##_broadcast_right((int16_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_sl##op_name##_broadcast_left((int16_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT16 && bDataType == DATA_TYPE_FLOAT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_sf##op_name((int16_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_sf##op_name(*(int16_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_sf##op_name((int16_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_sf##op_name##_broadcast_right((int16_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_sf##op_name##_broadcast_left((int16_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT16 && bDataType == DATA_TYPE_FLOAT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_sd##op_name((int16_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_sd##op_name(*(int16_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_sd##op_name((int16_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_sd##op_name##_broadcast_right((int16_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_sd##op_name##_broadcast_left((int16_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT32 && bDataType == DATA_TYPE_INT8) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ib##op_name((int32_t *) aPtr, *(int8_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ib##op_name(*(int32_t *) aPtr, (int8_t *) bPtr, (int32_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ib##op_name((int32_t *) aPtr, (int8_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ib##op_name##_broadcast_right((int32_t *) aPtr, (int8_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ib##op_name##_broadcast_left((int32_t *) aPtr, (int8_t *) bPtr, (int32_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT32 && bDataType == DATA_TYPE_INT16) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_is##op_name((int32_t *) aPtr, *(int16_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_is##op_name(*(int32_t *) aPtr, (int16_t *) bPtr, (int32_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_is##op_name((int32_t *) aPtr, (int16_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_is##op_name##_broadcast_right((int32_t *) aPtr, (int16_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_is##op_name##_broadcast_left((int32_t *) aPtr, (int16_t *) bPtr, (int32_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT32 && bDataType == DATA_TYPE_INT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ii##op_name((int32_t *) aPtr, *(int32_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ii##op_name(*(int32_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ii##op_name((int32_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ii##op_name##_broadcast_right((int32_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ii##op_name##_broadcast_left((int32_t *) aPtr, (int32_t *) bPtr, (int32_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT32 && bDataType == DATA_TYPE_INT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_il##op_name((int32_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_il##op_name(*(int32_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_il##op_name((int32_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_il##op_name##_broadcast_right((int32_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_il##op_name##_broadcast_left((int32_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT32 && bDataType == DATA_TYPE_FLOAT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_if##op_name((int32_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_if##op_name(*(int32_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_if##op_name((int32_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_if##op_name##_broadcast_right((int32_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_if##op_name##_broadcast_left((int32_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT32 && bDataType == DATA_TYPE_FLOAT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_id##op_name((int32_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_id##op_name(*(int32_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_id##op_name((int32_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_id##op_name##_broadcast_right((int32_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_id##op_name##_broadcast_left((int32_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT64 && bDataType == DATA_TYPE_INT8) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_lb##op_name((int64_t *) aPtr, *(int8_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_lb##op_name(*(int64_t *) aPtr, (int8_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_lb##op_name((int64_t *) aPtr, (int8_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_lb##op_name##_broadcast_right((int64_t *) aPtr, (int8_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_lb##op_name##_broadcast_left((int64_t *) aPtr, (int8_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT64 && bDataType == DATA_TYPE_INT16) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ls##op_name((int64_t *) aPtr, *(int16_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ls##op_name(*(int64_t *) aPtr, (int16_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ls##op_name((int64_t *) aPtr, (int16_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ls##op_name##_broadcast_right((int64_t *) aPtr, (int16_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ls##op_name##_broadcast_left((int64_t *) aPtr, (int16_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT64 && bDataType == DATA_TYPE_INT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_li##op_name((int64_t *) aPtr, *(int32_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_li##op_name(*(int64_t *) aPtr, (int32_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_li##op_name((int64_t *) aPtr, (int32_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_li##op_name##_broadcast_right((int64_t *) aPtr, (int32_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_li##op_name##_broadcast_left((int64_t *) aPtr, (int32_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT64 && bDataType == DATA_TYPE_INT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ll##op_name((int64_t *) aPtr, *(int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ll##op_name(*(int64_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ll##op_name((int64_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ll##op_name##_broadcast_right((int64_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ll##op_name##_broadcast_left((int64_t *) aPtr, (int64_t *) bPtr, (int64_t *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT64 && bDataType == DATA_TYPE_FLOAT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_lf##op_name((int64_t *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_lf##op_name(*(int64_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_lf##op_name((int64_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_lf##op_name##_broadcast_right((int64_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_lf##op_name##_broadcast_left((int64_t *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_INT64 && bDataType == DATA_TYPE_FLOAT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ld##op_name((int64_t *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ld##op_name(*(int64_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ld##op_name((int64_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ld##op_name##_broadcast_right((int64_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ld##op_name##_broadcast_left((int64_t *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT32 && bDataType == DATA_TYPE_INT8) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_fb##op_name((float *) aPtr, *(int8_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_fb##op_name(*(float *) aPtr, (int8_t *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_fb##op_name((float *) aPtr, (int8_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_fb##op_name##_broadcast_right((float *) aPtr, (int8_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_fb##op_name##_broadcast_left((float *) aPtr, (int8_t *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT32 && bDataType == DATA_TYPE_INT16) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_fs##op_name((float *) aPtr, *(int16_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_fs##op_name(*(float *) aPtr, (int16_t *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_fs##op_name((float *) aPtr, (int16_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_fs##op_name##_broadcast_right((float *) aPtr, (int16_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_fs##op_name##_broadcast_left((float *) aPtr, (int16_t *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT32 && bDataType == DATA_TYPE_INT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_fi##op_name((float *) aPtr, *(int32_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_fi##op_name(*(float *) aPtr, (int32_t *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_fi##op_name((float *) aPtr, (int32_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_fi##op_name##_broadcast_right((float *) aPtr, (int32_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_fi##op_name##_broadcast_left((float *) aPtr, (int32_t *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT32 && bDataType == DATA_TYPE_INT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_fl##op_name((float *) aPtr, *(int64_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_fl##op_name(*(float *) aPtr, (int64_t *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_fl##op_name((float *) aPtr, (int64_t *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_fl##op_name##_broadcast_right((float *) aPtr, (int64_t *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_fl##op_name##_broadcast_left((float *) aPtr, (int64_t *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT32 && bDataType == DATA_TYPE_FLOAT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ff##op_name((float *) aPtr, *(float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ff##op_name(*(float *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ff##op_name((float *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ff##op_name##_broadcast_right((float *) aPtr, (float *) bPtr, (float *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ff##op_name##_broadcast_left((float *) aPtr, (float *) bPtr, (float *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT32 && bDataType == DATA_TYPE_FLOAT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_fd##op_name((float *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_fd##op_name(*(float *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_fd##op_name((float *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_fd##op_name##_broadcast_right((float *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_fd##op_name##_broadcast_left((float *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT64 && bDataType == DATA_TYPE_INT8) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_db##op_name((double *) aPtr, *(int8_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_db##op_name(*(double *) aPtr, (int8_t *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_db##op_name((double *) aPtr, (int8_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_db##op_name##_broadcast_right((double *) aPtr, (int8_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_db##op_name##_broadcast_left((double *) aPtr, (int8_t *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT64 && bDataType == DATA_TYPE_INT16) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_ds##op_name((double *) aPtr, *(int16_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_ds##op_name(*(double *) aPtr, (int16_t *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_ds##op_name((double *) aPtr, (int16_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_ds##op_name##_broadcast_right((double *) aPtr, (int16_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_ds##op_name##_broadcast_left((double *) aPtr, (int16_t *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT64 && bDataType == DATA_TYPE_INT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_di##op_name((double *) aPtr, *(int32_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_di##op_name(*(double *) aPtr, (int32_t *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_di##op_name((double *) aPtr, (int32_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_di##op_name##_broadcast_right((double *) aPtr, (int32_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_di##op_name##_broadcast_left((double *) aPtr, (int32_t *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT64 && bDataType == DATA_TYPE_INT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_dl##op_name((double *) aPtr, *(int64_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_dl##op_name(*(double *) aPtr, (int64_t *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_dl##op_name((double *) aPtr, (int64_t *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_dl##op_name##_broadcast_right((double *) aPtr, (int64_t *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_dl##op_name##_broadcast_left((double *) aPtr, (int64_t *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT64 && bDataType == DATA_TYPE_FLOAT32) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_df##op_name((double *) aPtr, *(float *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_df##op_name(*(double *) aPtr, (float *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_df##op_name((double *) aPtr, (float *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_df##op_name##_broadcast_right((double *) aPtr, (float *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_df##op_name##_broadcast_left((double *) aPtr, (float *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else if (aDataType == DATA_TYPE_FLOAT64 && bDataType == DATA_TYPE_FLOAT64) {\
        if (nElementsC < std::max(nElementsA, nElementsB)) {\
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                             "C array is too small");\
        }\
        if (nElementsB == 1) {\
            tblas_tensor_dd##op_name((double *) aPtr, *(double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA == 1) {\
            tblas_tensor_dd##op_name(*(double *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB);\
        } else if (nElementsA == nElementsB) {\
            tblas_tensor_dd##op_name((double *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA);\
        } else if (nElementsA > nElementsB) {\
            tblas_tensor_dd##op_name##_broadcast_right((double *) aPtr, (double *) bPtr, (double *) cPtr, nElementsA, nElementsB);\
        } else if (nElementsB > nElementsA) {\
            tblas_tensor_dd##op_name##_broadcast_left((double *) aPtr, (double *) bPtr, (double *) cPtr, nElementsB, nElementsA);\
        }\
    } else {\
        jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"),\
                         "Unsupported data type");\
    }\
}
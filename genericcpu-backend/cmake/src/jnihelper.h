#pragma once

#include <jni.h>

#define OPERATION_JNI_WRAPPER(jniMethodName, operationFunction) \
JNIEXPORT void JNICALL jniMethodName(JNIEnv *jniEnv, jclass,\
                                                                          jlong aPtr, jlongArray shapeAArr,\
                                                                          jlongArray stridesAArr, jint dataTypeA,\
                                                                          jlong bPtr, jlongArray shapeBArr,\
                                                                          jlongArray stridesBArr, jint dataTypeB,\
                                                                          jlong cPtr, jlongArray shapeCArr, jlongArray stridesCArr,\
                                                                          jint dateTypeC) {\
    auto nDimsA = (size_t) jniEnv->GetArrayLength(shapeAArr);\
    auto nDimsB = (size_t) jniEnv->GetArrayLength(shapeBArr);\
    auto nDimsC = (size_t) jniEnv->GetArrayLength(shapeCArr);\
    auto *shapeA = new size_t[nDimsA];\
    auto *shapeB = new size_t[nDimsB];\
    auto *shapeC = new size_t[nDimsC];\
    auto *stridesA = new size_t[nDimsA];\
    auto *stridesB = new size_t[nDimsB];\
    auto *stridesC = new size_t[nDimsC];\
    {\
        auto *shapeALongArray = jniEnv->GetLongArrayElements(shapeAArr, nullptr);\
        auto *shapeBLongArray = jniEnv->GetLongArrayElements(shapeBArr, nullptr);\
        auto *shapeCLongArray = jniEnv->GetLongArrayElements(shapeCArr, nullptr);\
        for (size_t i = 0; i < nDimsA; i++) {\
            shapeA[i] = (size_t) shapeALongArray[i];\
        }\
        for (size_t i = 0; i < nDimsB; i++) {\
            shapeB[i] = (size_t) shapeBLongArray[i];\
        }\
        for (size_t i = 0; i < nDimsC; i++) {\
            shapeC[i] = (size_t) shapeCLongArray[i];\
        }\
        jniEnv->ReleaseLongArrayElements(shapeAArr, shapeALongArray, JNI_ABORT);\
        jniEnv->ReleaseLongArrayElements(shapeBArr, shapeBLongArray, JNI_ABORT);\
        jniEnv->ReleaseLongArrayElements(shapeCArr, shapeCLongArray, JNI_ABORT);\
        auto *stridesALongArray = jniEnv->GetLongArrayElements(stridesAArr, nullptr);\
        auto *stridesBLongArray = jniEnv->GetLongArrayElements(stridesBArr, nullptr);\
        auto *stridesCLongArray = jniEnv->GetLongArrayElements(stridesCArr, nullptr);\
        for (size_t i = 0; i < nDimsA; i++) {\
            stridesA[i] = (size_t) stridesALongArray[i];\
        }\
        for (size_t i = 0; i < nDimsB; i++) {\
            stridesB[i] = (size_t) stridesBLongArray[i];\
        }\
        for (size_t i = 0; i < nDimsC; i++) {\
            stridesC[i] = (size_t) stridesCLongArray[i];\
        }\
        jniEnv->ReleaseLongArrayElements(stridesAArr, stridesALongArray, JNI_ABORT);\
        jniEnv->ReleaseLongArrayElements(stridesBArr, stridesBLongArray, JNI_ABORT);\
        jniEnv->ReleaseLongArrayElements(stridesCArr, stridesCLongArray, JNI_ABORT);\
    }\
    if (dataTypeA == DATA_TYPE_INT8 && dataTypeB == DATA_TYPE_INT8) {\
        auto *a = (int8_t *) aPtr;\
        auto *b = (int8_t *) bPtr;\
        auto *c = (int8_t *) cPtr;\
        operationFunction(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC);\
    } else if (dataTypeA == DATA_TYPE_INT16 && dataTypeB == DATA_TYPE_INT16) {\
        auto *a = (int16_t *) aPtr;\
        auto *b = (int16_t *) bPtr;\
        auto *c = (int16_t *) cPtr;\
        operationFunction(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC);\
    } else if (dataTypeA == DATA_TYPE_INT32 && dataTypeB == DATA_TYPE_INT32) {\
        auto *a = (int32_t *) aPtr;\
        auto *b = (int32_t *) bPtr;\
        auto *c = (int32_t *) cPtr;\
        operationFunction(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC);\
    } else if (dataTypeA == DATA_TYPE_INT64 && dataTypeB == DATA_TYPE_INT64) {\
        auto *a = (int64_t *) aPtr;\
        auto *b = (int64_t *) bPtr;\
        auto *c = (int64_t *) cPtr;\
        operationFunction(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC);\
    } else if (dataTypeA == DATA_TYPE_FLOAT && dataTypeB == DATA_TYPE_FLOAT) {\
        auto *a = (float *) aPtr;\
        auto *b = (float *) bPtr;\
        auto *c = (float *) cPtr;\
        operationFunction(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC);\
    } else if (dataTypeA == DATA_TYPE_DOUBLE && dataTypeB == DATA_TYPE_DOUBLE) {\
        auto *a = (double *) aPtr;\
        auto *b = (double *) bPtr;\
        auto *c = (double *) cPtr;\
        operationFunction(a, b, c, shapeA, stridesA, nDimsA, shapeB, stridesB, nDimsB, shapeC, stridesC, nDimsC);\
    } else {\
        jclass exceptionClass = jniEnv->FindClass("java/lang/IllegalArgumentException");\
        jniEnv->ThrowNew(exceptionClass, "Unsupported data type");\
    }\
    delete[] shapeA;\
    delete[] shapeB;\
}
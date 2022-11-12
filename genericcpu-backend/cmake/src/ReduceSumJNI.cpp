#include <stdexcept>
#include "ReduceSumJNI.h"
#include "jnidatatypes.h"
#include <reducesum.h>

void Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ReduceSumJNI_nreduceSum(JNIEnv *jniEnv, jclass, jlong aPtr,
                                                                                 jlong cPtr, jint dataType,
                                                                                 jlongArray shapeA, jlongArray stridesA,
                                                                                 jlongArray shapeC, jlongArray stridesC,
                                                                                 jlong dimension, jboolean keepDims) {
    size_t nDimsA = jniEnv->GetArrayLength(shapeA);
    size_t nDimsC = jniEnv->GetArrayLength(shapeC);
    if (nDimsA != jniEnv->GetArrayLength(stridesA)) {
        jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"), "shapeA and stridesA must have the same length");
    }
    if (nDimsC != jniEnv->GetArrayLength(stridesC)) {
        jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"), "shapeC and stridesC must have the same length");
    }
    auto *shapeA_ = new size_t[nDimsA];
    auto *stridesA_ = new size_t[nDimsA];
    auto *shapeC_ = new size_t[nDimsC];
    auto *stridesC_ = new size_t[nDimsC];
    jniEnv->GetLongArrayRegion(shapeA, 0, static_cast<jsize>(nDimsA), reinterpret_cast<jlong *>(shapeA_));
    jniEnv->GetLongArrayRegion(stridesA, 0, static_cast<jsize>(nDimsA), reinterpret_cast<jlong *>(stridesA_));
    jniEnv->GetLongArrayRegion(shapeC, 0, static_cast<jsize>(nDimsC), reinterpret_cast<jlong *>(shapeC_));
    jniEnv->GetLongArrayRegion(stridesC, 0, static_cast<jsize>(nDimsC), reinterpret_cast<jlong *>(stridesC_));

    switch (dataType) {
        case DATA_TYPE_INT8:
            tblas_reducesum(reinterpret_cast<int8_t *>(aPtr), reinterpret_cast<int8_t *>(cPtr),
                            shapeA_, stridesA_, nDimsA,
                            shapeC_, stridesC_, nDimsC,
                            static_cast<int64_t>(dimension), static_cast<bool>(keepDims));
            break;
        case DATA_TYPE_INT16:
            tblas_reducesum(reinterpret_cast<int16_t *>(aPtr), reinterpret_cast<int16_t *>(cPtr),
                            shapeA_, stridesA_, nDimsA,
                            shapeC_, stridesC_, nDimsC,
                            static_cast<int64_t>(dimension), static_cast<bool>(keepDims));
            break;
        case DATA_TYPE_INT32:
            tblas_reducesum(reinterpret_cast<int32_t *>(aPtr), reinterpret_cast<int32_t *>(cPtr),
                            shapeA_, stridesA_, nDimsA,
                            shapeC_, stridesC_, nDimsC,
                            static_cast<int64_t>(dimension), static_cast<bool>(keepDims));
            break;
        case DATA_TYPE_INT64:
            tblas_reducesum(reinterpret_cast<int64_t *>(aPtr), reinterpret_cast<int64_t *>(cPtr),
                            shapeA_, stridesA_, nDimsA,
                            shapeC_, stridesC_, nDimsC,
                            static_cast<int64_t>(dimension), static_cast<bool>(keepDims));
            break;
        case DATA_TYPE_FLOAT32:
            tblas_reducesum(reinterpret_cast<float *>(aPtr), reinterpret_cast<float *>(cPtr),
                            shapeA_, stridesA_, nDimsA,
                            shapeC_, stridesC_, nDimsC,
                            static_cast<int64_t>(dimension), static_cast<bool>(keepDims));
            break;
        case DATA_TYPE_FLOAT64:
            tblas_reducesum(reinterpret_cast<double *>(aPtr), reinterpret_cast<double *>(cPtr),
                            shapeA_, stridesA_, nDimsA,
                            shapeC_, stridesC_, nDimsC,
                            static_cast<int64_t>(dimension), static_cast<bool>(keepDims));
            break;
        default:
            jniEnv->ThrowNew(jniEnv->FindClass("java/lang/IllegalArgumentException"), "Unsupported data type");
            break;
    }

    delete[] shapeA_;
    delete[] stridesA_;
    delete[] shapeC_;
    delete[] stridesC_;
}

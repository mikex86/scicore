#include "ArgmaxJNI.h"
#include "jnidatatypes.h"
#include <stdexcept>
#include <cstdint>
#include <argmax.h>

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ArgmaxJNI_nargmax(JNIEnv *jniEnv, jclass,
                                                                      jlong aPtr, jlongArray shapeAArr, jlongArray stridesAArr,
                                                                      jlong cPtr, jlongArray shapeCArr, jlongArray stridesCArr,
                                                                      jint dataType,
                                                                      jint dimension) {

    size_t nDimsA = jniEnv->GetArrayLength(shapeAArr);
    size_t nDimsC = jniEnv->GetArrayLength(shapeCArr);

    auto *shapeA = new size_t[nDimsA];
    auto *stridesA = new size_t[nDimsA];

    auto *shapeC = new size_t[nDimsC];
    auto *stridesC = new size_t[nDimsC];

    jniEnv->GetLongArrayRegion(shapeAArr, 0, static_cast<jsize>(nDimsA), reinterpret_cast<jlong *>(shapeA));
    jniEnv->GetLongArrayRegion(stridesAArr, 0, static_cast<jsize>(nDimsA), reinterpret_cast<jlong *>(stridesA));

    jniEnv->GetLongArrayRegion(shapeCArr, 0, static_cast<jsize>(nDimsC), reinterpret_cast<jlong *>(shapeC));
    jniEnv->GetLongArrayRegion(stridesCArr, 0, static_cast<jsize>(nDimsC), reinterpret_cast<jlong *>(stridesC));


    switch (dataType) {
        case DATA_TYPE_INT8: {
            tblas_argmax(reinterpret_cast<int8_t *>(aPtr), reinterpret_cast<uint64_t *>(cPtr),
                         shapeA, stridesA, nDimsA,
                         shapeC, stridesC, nDimsC,
                         static_cast<int64_t>(dimension));
            break;
        }
        case DATA_TYPE_INT16: {
            tblas_argmax(reinterpret_cast<int16_t *>(aPtr), reinterpret_cast<uint64_t *>(cPtr),
                         shapeA, stridesA, nDimsA,
                         shapeC, stridesC, nDimsC,
                         static_cast<int64_t>(dimension));
            break;
        }
        case DATA_TYPE_INT32: {
            tblas_argmax(reinterpret_cast<int32_t *>(aPtr), reinterpret_cast<uint64_t *>(cPtr),
                         shapeA, stridesA, nDimsA,
                         shapeC, stridesC, nDimsC,
                         static_cast<int64_t>(dimension));
            break;
        }
        case DATA_TYPE_INT64: {
            tblas_argmax(reinterpret_cast<int64_t *>(aPtr), reinterpret_cast<uint64_t *>(cPtr),
                         shapeA, stridesA, nDimsA,
                         shapeC, stridesC, nDimsC,
                         static_cast<int64_t>(dimension));
            break;
        }
        case DATA_TYPE_FLOAT32: {
            tblas_argmax(reinterpret_cast<float *>(aPtr), reinterpret_cast<uint64_t *>(cPtr),
                         shapeA, stridesA, nDimsA,
                         shapeC, stridesC, nDimsC,
                         static_cast<int64_t>(dimension));
            break;
        }
        case DATA_TYPE_FLOAT64: {
            tblas_argmax(reinterpret_cast<double *>(aPtr), reinterpret_cast<uint64_t *>(cPtr),
                         shapeA, stridesA, nDimsA,
                         shapeC, stridesC, nDimsC,
                         static_cast<int64_t>(dimension));
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type");
    }


}

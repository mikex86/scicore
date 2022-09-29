#include "DivideJNI.h"
#include "divide.h"
#include "jnihelper.h"

static void throwDivideByZeroException(JNIEnv * jniEnv) {
    jclass exceptionClass = jniEnv->FindClass("java/lang/ArithmeticException");
    jniEnv->ThrowNew(exceptionClass, "Divide by zero");
}

BINARY_OP_JNI_WRAPPER_FUNC_FOR_ALL_TYPES_ALL_VARIANTS(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_DivideJNI_ndivide, divide, {
    switch (bDataType) {
        case DATA_TYPE_INT8: {
            auto *b = (int8_t *) bPtr;
            for (int i = 0; i < nElementsB; i++) {
                if (b[i] == 0) {
                    throwDivideByZeroException(jniEnv);
                    return;
                }
            }
            break;
        }
        case DATA_TYPE_INT16: {
            auto *b = (int16_t *) bPtr;
            for (int i = 0; i < nElementsB; i++) {
                if (b[i] == 0) {
                    throwDivideByZeroException(jniEnv);
                    return;
                }
            }
            break;
        }
        case DATA_TYPE_INT32: {
            auto *b = (int32_t *) bPtr;
            for (int i = 0; i < nElementsB; i++) {
                if (b[i] == 0) {
                    throwDivideByZeroException(jniEnv);
                    return;
                }
            }
            break;
        }
        case DATA_TYPE_INT64: {
            auto *b = (int64_t *) bPtr;
            for (int i = 0; i < nElementsB; i++) {
                if (b[i] == 0) {
                    throwDivideByZeroException(jniEnv);
                    return;
                }
            }
            break;
        }
        default: {
            break;
        }
        // Allowed for floats
    }
});
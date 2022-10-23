#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ReduceSumJNI_nreduceSum(JNIEnv *jniEnv, jclass,
                                                                          jlong aPtr,
                                                                          jlong cPtr,
                                                                          jint dataType,
                                                                          jlongArray shapeA, jlongArray stridesA,
                                                                          jlongArray shapeC, jlongArray stridesC,
                                                                          jlong dimension, jboolean keepDims);
#ifdef __cplusplus
}
#endif

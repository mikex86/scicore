#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ArgmaxJNI_nargmax(JNIEnv *jniEnv, jclass,
                                                                      jlong aPtr, jlongArray shapeAArr,
                                                                      jlongArray stridesAArr,
                                                                      jlong cPtr, jlongArray shapeCArr,
                                                                      jlongArray stridesCArr,
                                                                      jint dataType,
                                                                      jint dimension);

#ifdef __cplusplus
}
#endif

#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MinusJNI_nminus(JNIEnv *jniEnv, jclass,
                                                                          jlong aPtr, jlongArray shapeAArr,
                                                                          jlongArray stridesAArr, jint dataTypeA,
                                                                          jlong bPtr, jlongArray shapeBArr,
                                                                          jlongArray stridesBArr, jint dataTypeB,
                                                                          jlong cPtr, jlongArray shapeCArr,
                                                                          jlongArray stridesCArr,
                                                                          jint dateTypeC);

#ifdef __cplusplus
}
#endif

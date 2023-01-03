#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {

#endif
JNIEXPORT void
JNICALL Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MatmulJNI_nmatmul(JNIEnv *, jclass, jint transa,
                                                                              jint transb,
                                                                              jint layout,
                                                                              jint m, jint n, jint k,
                                                                              jlong alphaPtr,
                                                                              jlong aPtr,
                                                                              jint aType,
                                                                              jint lda,
                                                                              jlong betaPtr, jlong bPtr,
                                                                              jint bType,
                                                                              jint ldb,
                                                                              jlong cPtr,
                                                                              jint cType,
                                                                              jint ldc);

#ifdef __cplusplus
}
#endif
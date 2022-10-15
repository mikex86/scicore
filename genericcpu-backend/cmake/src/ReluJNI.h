#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     me_mikex86_scicore_backend_impl_genericcpu_jni_ReluJNI
 * Method:    nexp
 * Signature: (JJIJJ)V
 */
JNIEXPORT void JNICALL Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ReluJNI_nrelu
        (JNIEnv *, jclass, jlong, jlong, jlong, jint);

JNIEXPORT void JNICALL Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ReluJNI_nreluGradients
        (JNIEnv *, jclass, jlong, jlong, jlong, jint);

#ifdef __cplusplus
}
#endif

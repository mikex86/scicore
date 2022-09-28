#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MultiplyJNI_nmultiply
  (JNIEnv *, jclass, jlong, jint, jlong, jlong, jint, jlong, jlong, jlong);

#ifdef __cplusplus
}
#endif
#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_me_mikex86_scicore_backend_impl_genericcpu_jni_TanhJNI_ntanh
        (JNIEnv *, jclass, jlong, jlong, jlong, jint);

#ifdef __cplusplus
}
#endif

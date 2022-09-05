#include <jni.h>

#ifndef _Included_me_mikex86_scicore_backend_impl_genericcpu_op_GenCPUMatMulOp
#define _Included_me_mikex86_scicore_backend_impl_genericcpu_op_GenCPUMatMulOp
#ifdef __cplusplus
extern "C" {

#endif
JNIEXPORT void JNICALL Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MatmulJNI_matmul(JNIEnv *, jclass, jint transa, jint transb,
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
#endif

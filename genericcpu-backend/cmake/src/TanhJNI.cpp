#include "TanhJNI.h"
#include <tanh.h>

#include "jnihelper.h"

UNARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_TanhJNI_ntanh, tblas_tanh);

UNARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_TanhJNI_ntanhGradients, tblas_tanh_gradients);
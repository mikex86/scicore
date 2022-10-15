#include "ReluJNI.h"
#include <relu.h>

#include "jnihelper.h"

UNARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ReluJNI_nrelu, tblas_relu);

UNARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ReluJNI_nreluGradients, tblas_relu_gradients);
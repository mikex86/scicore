#include "ExpJNI.h"
#include <exp.h>

#include "jnihelper.h"

UNARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_ExpJNI_nexp, tblas_exp)
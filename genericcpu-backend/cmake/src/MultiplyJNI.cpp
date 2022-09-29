#include "MultiplyJNI.h"

#include "multiply.h"
#include "jnihelper.h"

BINARY_OP_JNI_WRAPPER_FUNC_FOR_ALL_TYPES_ALL_VARIANTS(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MultiplyJNI_nmultiply, mul,);
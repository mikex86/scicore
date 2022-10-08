#include "MultiplyJNI.h"
#include "jnihelper.h"
#include <multiply.h>
#include <cstdint>

BINARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MultiplyJNI_nmultiply, tblas_multiply)
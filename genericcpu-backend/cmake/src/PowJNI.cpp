#include "PowJNI.h"
#include "jnihelper.h"
#include <pow.h>
#include <cstdint>

BINARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_PowJNI_npow, tblas_pow)
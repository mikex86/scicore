#include "MinusJNI.h"
#include "jnihelper.h"
#include <minus.h>
#include <cstdint>

BINARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MinusJNI_nminus, tblas_minus)
#include "DivideJNI.h"
#include "jnihelper.h"
#include <divide.h>
#include <cstdint>

BINARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_DivideJNI_ndivide, tblas_divide)
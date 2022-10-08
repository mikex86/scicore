#include "MinusJNI.h"
#include "jnihelper.h"
#include <minus.h>
#include <cstdint>

#define DATA_TYPE_INT8 1
#define DATA_TYPE_INT16 2
#define DATA_TYPE_INT32 3
#define DATA_TYPE_INT64 4
#define DATA_TYPE_FLOAT 5
#define DATA_TYPE_DOUBLE 6

OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_MinusJNI_nminus, tblas_minus)
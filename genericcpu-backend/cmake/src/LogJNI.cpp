#include "LogJNI.h"
#include <log.h>

#include "jnihelper.h"

UNARY_OPERATION_JNI_WRAPPER(Java_me_mikex86_scicore_backend_impl_genericcpu_jni_LogJNI_nlog, tblas_log);
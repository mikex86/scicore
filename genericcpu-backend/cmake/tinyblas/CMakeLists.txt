project(tinyblas)


# Enable -O3, -march=native and -ffast-math
if (MSVC)
    # if MSVC
    set(CMAKE_CXX_FLAGS "/O2 /arch:AVX2 /fp:fast /GL")
else ()
    if (APPLE)
        set(CMAKE_CXX_FLAGS "-O3 -ffast-math")
    else ()
        set(CMAKE_CXX_FLAGS "-O3 -march=native -ffast-math")
    endif ()
endif ()

# use -fPIC on Linux
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()

# Check if arm neon is available
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <arm_neon.h>
int main() {
    int32x4_t a = vdupq_n_s32(0);
    return 0;
}
" HAS_ARM_NEON)
if (HAS_ARM_NEON)
    add_subdirectory(libraries/neon_mathfun)
endif ()

add_library(tinyblas STATIC src/matmul.cpp src/multiply.cpp src/plus.cpp src/minus.cpp src/divide.cpp src/exp.cpp src/pow.cpp src/relu.cpp src/reducesum.cpp src/argmax.cpp src/sigmoid.cpp src/tanh.cpp src/log.cpp)
target_include_directories(tinyblas PUBLIC include)

if (HAS_ARM_NEON)
    target_link_libraries(tinyblas PRIVATE neon_mathfun)
endif ()
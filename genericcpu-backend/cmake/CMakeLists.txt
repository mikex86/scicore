cmake_minimum_required(VERSION 3.22)
project(scicore_genericcpu)

set(CMAKE_CXX_STANDARD 14)

add_library(scicore_genericcpu SHARED
        src/MatmulJNI.cpp src/MultiplyJNI.cpp src/PlusJNI.cpp src/MinusJNI.cpp src/DivideJNI.cpp
        src/ExpJNI.cpp src/PowJNI.cpp src/ReluJNI.cpp src/SigmoidJNI.cpp
        src/ReduceSumJNI.cpp src/ArgmaxJNI.cpp)

# Find JNI
set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_JVM_LIBRARY NotNeeded)
#set(JAVA_INCLUDE_PATH2 NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)
target_include_directories(scicore_genericcpu PRIVATE ${JNI_INCLUDE_DIRS})

# Use MKL if available
# Use static linking
set(MKL_LINK static)
set(MKL_INTERFACE ilp64) # Use 64-bit integers
find_package(MKL CONFIG)

if (APPLE)
    message(STATUS "Using Apple Accelerate as BLAS/LAPACK backend")
    # Link vecLib on macOS
    target_link_libraries(scicore_genericcpu PRIVATE "-framework Accelerate")
elseif (MKL_FOUND)
    message(STATUS "Using MKL as BLAS/LAPACK backend")
    target_compile_definitions(scicore_genericcpu PUBLIC USE_MKL) # Define USE_MKL macro
    target_compile_definitions(scicore_genericcpu PRIVATE MKL_ILP64) # Make MKL headers use 64-bit integers
    target_compile_options(scicore_genericcpu PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(scicore_genericcpu PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(scicore_genericcpu PUBLIC MKL::MKL)
else ()
    message(STATUS "Using tinyBLAS as BLAS/LAPACK backend")
endif ()

# Tinyblas
add_subdirectory(tinyblas)
target_link_libraries(scicore_genericcpu PRIVATE tinyblas)
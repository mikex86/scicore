cmake_minimum_required(VERSION 3.22)
project(scicore_genericcpu)

set(CMAKE_CXX_STANDARD 14)

add_library(scicore_genericcpu SHARED src/matmul.cpp)

# Find JNI
find_package(JNI REQUIRED)
target_include_directories(scicore_genericcpu PUBLIC ${JNI_INCLUDE_DIRS})

if (APPLE)
    # Link vecLib on macOS
    target_link_libraries(scicore_genericcpu PRIVATE "-framework Accelerate")
else ()
    # Use OpenBLAS on all other platforms
    # build static library
    option(BUILD_STATIC_LIBS "Build static library" OFF)
    add_subdirectory(OpenBLAS)
    target_include_directories(scicore_genericcpu PRIVATE "${CMAKE_BINARY_DIR}/OpenBLAS")
    target_link_libraries(scicore_genericcpu PRIVATE openblas)
    target_include_directories(scicore_genericcpu PRIVATE OpenBLAS)
endif ()
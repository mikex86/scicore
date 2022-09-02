cmake_minimum_required(VERSION 3.22)
project(scicore_genericcpu)

set(CMAKE_CXX_STANDARD 14)

add_library(scicore_genericcpu SHARED src/matmul.cpp)


# TODO: Better JNI locating
target_include_directories(scicore_genericcpu PRIVATE "$ENV{JAVA_INCLUDE_PATH}")

# Link vecLib on macOS
if (APPLE)
    target_link_libraries(scicore_genericcpu PRIVATE "-framework Accelerate")
else ()
endif ()
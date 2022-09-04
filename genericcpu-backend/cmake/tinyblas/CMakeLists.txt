project(tinyblas)

# Enable -O3, -march=native and -ffast-math
set(CMAKE_CXX_FLAGS "/O2 /arch:AVX2 /fp:fast /GL")

add_library(tinyblas STATIC src/matmul.cpp)
target_include_directories(tinyblas PUBLIC include)
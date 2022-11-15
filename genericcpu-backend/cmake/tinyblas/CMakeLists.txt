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

add_library(tinyblas STATIC src/matmul.cpp src/multiply.cpp src/plus.cpp src/minus.cpp src/divide.cpp src/exp.cpp src/pow.cpp src/relu.cpp src/reducesum.cpp src/argmax.cpp src/sigmoid.cpp src/tanh.cpp src/log.cpp)
target_include_directories(tinyblas PUBLIC include)
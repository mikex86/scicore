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

add_library(tinyblas STATIC src/matmul.cpp src/tensormul.cpp)
target_include_directories(tinyblas PUBLIC include)
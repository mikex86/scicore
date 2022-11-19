project(neon_mathfun)

add_library(neon_mathfun STATIC src/neon_mathfun.cpp)
target_include_directories(neon_mathfun PUBLIC include)
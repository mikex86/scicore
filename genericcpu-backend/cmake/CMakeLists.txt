cmake_minimum_required(VERSION 3.22)
project(scicore_genericcpu)

set(CMAKE_CXX_STANDARD 14)

add_library(scicore_genericcpu SHARED src/matmul.cpp)

# Find JNI
set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_JVM_LIBRARY NotNeeded)
#set(JAVA_INCLUDE_PATH2 NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)
target_include_directories(scicore_genericcpu PRIVATE ${JNI_INCLUDE_DIRS})

if (APPLE)
    # Link vecLib on macOS
    target_link_libraries(scicore_genericcpu PRIVATE "-framework Accelerate")
else ()
    # Use tinyblas on other platforms
    add_subdirectory(tinyblas)
    target_link_libraries(scicore_genericcpu PRIVATE tinyblas)
endif ()
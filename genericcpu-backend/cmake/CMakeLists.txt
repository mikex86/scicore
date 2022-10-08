cmake_minimum_required(VERSION 3.22)
project(scicore_genericcpu)

set(CMAKE_CXX_STANDARD 14)

add_library(scicore_genericcpu SHARED src/MatmulJNI.cpp src/MultiplyJNI.cpp src/PlusJNI.cpp src/MinusJNI.cpp src/DivideJNI.cpp src/ExpJNI.cpp src/PowJNI.cpp)

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
endif ()

# Tinyblas
add_subdirectory(tinyblas)
target_link_libraries(scicore_genericcpu PRIVATE tinyblas)
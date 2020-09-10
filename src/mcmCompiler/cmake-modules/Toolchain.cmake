#
# CMAKE Toolchain file used for cross compiling MCM Compiler for Linaro systems
#
# Instructions:
# 1) linaro GCC/G++ compiler needs to installed (from linaro.org)
# All library .h files need to be placed in 
#  /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/c++/7.4.1/lib
#
# 2) Use this file with command:
#    cmake -DCMAKE_TOOLCHAIN_FILE=./Toolchain.cmake ..
#
# NOTE! This build does NOT run the meta build and create generated files.
# Temporarily, run a standard build first then run this Cross Compile build

# specify system type (required)
SET(CMAKE_SYSTEM_NAME Linux)

# specify cross compiler for C and C++
set(TOOLCHAIN_PREFIX /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-)
SET(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)

# python libraries for 
SET(PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libpython3.6m.so)
SET(PYTHON_INCLUDE_DIR /usr/lib/python3.7/)

SET(CMAKE_SYSROOT /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc)

SET(CMAKE_FIND_ROOT_PATH 
    /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu
    /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc
    /opt/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib
)

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
# SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# put flags allowing dynamic symbols put into target
macro(replace_compile_options)
    # Replace compiler flags
    foreach(flag "-fvisibility=default" "-fvisibility=hidden" "-rdynamic" "-export-dynamic")
        string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    endforeach()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -rdynamic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -export-dynamic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -export-dynamic")
endmacro(replace_compile_options)

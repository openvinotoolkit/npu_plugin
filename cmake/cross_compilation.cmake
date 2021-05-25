#
# Copyright 2020 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

function(vpux_add_native_tool NATIVE_NAME NATIVE_SOURCE_DIR)
    if(NOT CMAKE_CROSSCOMPILING)
        set(${NATIVE_NAME}_COMMAND ${NATIVE_NAME} CACHE INTERNAL "" FORCE)
        set(${NATIVE_NAME}_TARGET ${NATIVE_NAME} CACHE INTERNAL "" FORCE)
        return()
    endif()

    set(options)
    set(oneValueArgs "EXEDIR")
    set(multiValueArgs "CMAKE_ARGS")
    cmake_parse_arguments(NATIVE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(NATIVE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/NATIVE/${NATIVE_NAME}")

    if(NOT DEFINED NATIVE_EXEDIR)
        set(NATIVE_EXEDIR ".")
    endif()

    if(CMAKE_CFG_INTDIR STREQUAL ".")
        set(NATIVE_CFGDIR ".")
    else()
        set(NATIVE_CFGDIR "Release")
    endif()

    set(${NATIVE_NAME}_COMMAND
        "${NATIVE_BINARY_DIR}/${NATIVE_EXEDIR}/${NATIVE_CFGDIR}/${NATIVE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
        CACHE INTERNAL "" FORCE
    )
    set(${NATIVE_NAME}_TARGET
        NATIVE_${NATIVE_NAME}
        CACHE INTERNAL "" FORCE
    )

    add_custom_command(
        OUTPUT ${NATIVE_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NATIVE_BINARY_DIR}
        COMMENT "[NATIVE] Creating ${NATIVE_BINARY_DIR} ..."
    )

    set(cmake_args -G ${CMAKE_GENERATOR})
    if(CMAKE_GENERATOR_TOOLSET)
        list(APPEND cmake_args -T ${CMAKE_GENERATOR_TOOLSET})
    endif()
    if(CMAKE_GENERATOR_PLATFORM)
        list(APPEND cmake_args -A ${CMAKE_GENERATOR_PLATFORM})
    endif()
    list(APPEND cmake_args -S ${NATIVE_SOURCE_DIR})
    list(APPEND cmake_args -B ${NATIVE_BINARY_DIR})
    if(CMAKE_BUILD_TYPE)
        list(APPEND cmake_args -D "CMAKE_BUILD_TYPE:STRING=Release")
    elseif(CMAKE_CONFIGURATION_TYPES)
        list(APPEND cmake_args -D "CMAKE_CONFIGURATION_TYPES:STRING=Release")
    endif()
    foreach(arg IN LISTS NATIVE_CMAKE_ARGS)
        list(APPEND cmake_args -D "${arg}")
    endforeach()

    add_custom_command(
        OUTPUT "${NATIVE_BINARY_DIR}/CMakeCache.txt"
        COMMAND ${CMAKE_COMMAND} ${cmake_args}
        DEPENDS ${NATIVE_BINARY_DIR} ${NATIVE_NAME}
        COMMENT "[NATIVE] Configuring ${NATIVE_NAME} ..."
    )

    add_custom_command(
        OUTPUT ${${NATIVE_NAME}_COMMAND}
        COMMAND ${CMAKE_COMMAND} --build ${NATIVE_BINARY_DIR} --config Release --target ${NATIVE_NAME}
        DEPENDS "${NATIVE_BINARY_DIR}/CMakeCache.txt"
        COMMENT "[NATIVE] Building ${NATIVE_NAME} ..."
    )

    add_custom_target(NATIVE_${NATIVE_NAME}
        DEPENDS ${${NATIVE_NAME}_COMMAND}
    )
endfunction()

function(vpux_add_crosscompile_project CROSSCOMPILE_NAME SOURCE_DIR TOOLCHAIN_FILE)
    if(CMAKE_CROSSCOMPILING)
        set(${CROSSCOMPILE_NAME}_COMMAND ${CROSSCOMPILE_NAME} CACHE INTERNAL "" FORCE)
        set(${CROSSCOMPILE_NAME}_TARGET ${CROSSCOMPILE_NAME} CACHE INTERNAL "" FORCE)
        return()
    endif()

    set(options)
    set(oneValueArgs "EXEDIR")
    set(multiValueArgs "CMAKE_ARGS")
    cmake_parse_arguments(CROSSCOMPILE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(CROSSCOMPILE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/CROSSCOMPILE/${CROSSCOMPILE_NAME}")

    if(NOT DEFINED CROSSCOMPILE_EXEDIR)
        set(CROSSCOMPILE_EXEDIR ".")
    endif()

    if(CMAKE_CFG_INTDIR STREQUAL ".")
        set(CROSSCOMPILE_CFGDIR ".")
    else()
        set(CROSSCOMPILE_CFGDIR "Release")
    endif()

    set(${CROSSCOMPILE_NAME}_COMMAND
            "${CROSSCOMPILE_BINARY_DIR}/${CROSSCOMPILE_EXEDIR}/${CROSSCOMPILE_CFGDIR}/${CROSSCOMPILE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
            CACHE INTERNAL "" FORCE
            )
    set(${CROSSCOMPILE_NAME}_TARGET
            CROSSCOMPILE_${CROSSCOMPILE_NAME}
            CACHE INTERNAL "" FORCE
            )

    add_custom_command(
            OUTPUT ${CROSSCOMPILE_BINARY_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CROSSCOMPILE_BINARY_DIR}
            COMMENT "[CROSSCOMPILE] Creating ${CROSSCOMPILE_BINARY_DIR} ..."
    )

    set(cmake_args -G ${CMAKE_GENERATOR})
    if(CMAKE_GENERATOR_TOOLSET)
        list(APPEND cmake_args -T ${CMAKE_GENERATOR_TOOLSET})
    endif()
    if(CMAKE_GENERATOR_PLATFORM)
        list(APPEND cmake_args -A ${CMAKE_GENERATOR_PLATFORM})
    endif()
    list(APPEND cmake_args -S ${SOURCE_DIR})
    list(APPEND cmake_args -B ${CROSSCOMPILE_BINARY_DIR})
    if(CMAKE_BUILD_TYPE)
        list(APPEND cmake_args -D "CMAKE_BUILD_TYPE:STRING=Release")
    elseif(CMAKE_CONFIGURATION_TYPES)
        list(APPEND cmake_args -D "CMAKE_CONFIGURATION_TYPES:STRING=Release")
    endif()
    list(APPEND cmake_args -D "CMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
    foreach(arg IN LISTS CROSSCOMPILE_CMAKE_ARGS)
        list(APPEND cmake_args -D "${arg}")
    endforeach()

    add_custom_command(
            OUTPUT "${CROSSCOMPILE_BINARY_DIR}/CMakeCache.txt"
            COMMAND ${CMAKE_COMMAND} ${cmake_args}
            DEPENDS ${CROSSCOMPILE_BINARY_DIR}
            COMMENT "[CROSSCOMPILE] Configuring ${CROSSCOMPILE_NAME} ..."
    )

    add_custom_target(CROSSCOMPILE_${CROSSCOMPILE_NAME} ALL
            COMMAND ${CMAKE_COMMAND} --build ${CROSSCOMPILE_BINARY_DIR} --config $<CONFIG>
            DEPENDS "${CROSSCOMPILE_BINARY_DIR}/CMakeCache.txt" ${CROSSCOMPILE_BINARY_DIR}
            COMMENT "[CROSSCOMPILE] Building ${CROSSCOMPILE_NAME} ..."
            )
endfunction()

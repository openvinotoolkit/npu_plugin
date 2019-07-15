# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8.11)

project(Samples)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set (IE_MAIN_SAMPLES_DIR "${IE_MAIN_SOURCE_DIR}/samples")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror") #treating warnings as errors
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
endif()

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
####################################

set (GFLAGS_IS_SUBPROJECT TRUE)
set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)

add_subdirectory(${IE_MAIN_SAMPLES_DIR}/thirdparty/gflags ${CMAKE_CURRENT_BINARY_DIR}/gflags)

function(add_samples_to_build)
    # check each passed sample subdirectory
    foreach (dir ${ARGN})
        if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
            # check if a subdirectory contains CMakeLists.txt. In this case we can build it.
            file(GLOB is_sample_dir "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
            if(is_sample_dir)
                # check if specified sample/demo is found.
                if (BUILD_SAMPLE_NAME)
                    list(FIND BUILD_SAMPLE_NAME ${dir} index)
                endif()
                if (index EQUAL -1)
                    message(STATUS "${dir} SKIPPED")
                else()
                    # Include subdirectory to the project.
                    add_subdirectory(${dir})
                endif()
            endif()
        endif()
    endforeach()
endfunction(add_samples_to_build)

include(CMakeParseArguments)

macro(ie_add_sample)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES OPENCV_DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    source_group("src" FILES ${IE_SAMPLES_SOURCES})
    if(IE_SAMPLES_HEADERS)
        source_group("include" FILES ${IE_SAMPLES_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_SAMPLE_NAME} ${IE_SAMPLE_SOURCES} ${IE_SAMPLES_HEADERS})

    target_include_directories(${IE_SAMPLE_NAME}
        PRIVATE
        "${IE_MAIN_SAMPLES_DIR}/common"
    )

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${InferenceEngine_LIBRARIES}
                                                    ${IE_SAMPLE_DEPENDENCIES} IE::ie_cpu_extension gflags)

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE pthread)

    # create global target with all samples / demo apps
    if(NOT TARGET ie_samples)
        add_custom_target(ie_samples ALL)
    endif()
    add_dependencies(ie_samples ${IE_SAMPLE_NAME})

endmacro()

# collect all samples subdirectories
file(GLOB samples_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
# skip building of unnecessary subdirectories
list(REMOVE_ITEM samples_dirs common thirdparty)
add_samples_to_build(${samples_dirs})

# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8.11)

project(Samples)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if (NOT(IE_MAIN_SOURCE_DIR))
    # in case if samples are built out of IE repo
    set (IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
else()
    # in case if samples are built from IE repo
    set (IE_MAIN_SAMPLES_DIR ${IE_MAIN_SOURCE_DIR})
endif()

if(NOT(UNIX))
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

if (WIN32)
    if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX") #treating warnings as errors
    endif ()

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings
    endif()
else()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror") #treating warnings as errors
    if (APPLE)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
        if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
        endif()
    endif()
endif()

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
####################################

if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

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

if(NOT TARGET gflags)
    find_package(gflags REQUIRED)
endif()

include(CMakeParseArguments)

macro(ie_add_sample)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES OPENCV_DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Find OpenCV components if exist
    if(IE_SAMPLE_OPENCV_DEPENDENCIES)
        find_package(OpenCV COMPONENTS ${IE_SAMPLE_OPENCV_DEPENDENCIES} QUIET)
        if(NOT OpenCV_FOUND)
            message(WARNING "OPENCV is disabled or not found, " ${IE_SAMPLE_NAME} " skipped")
            return()
        else()
            add_definitions(-DUSE_OPENCV)
        endif()
    endif()

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_SAMPLES_SOURCES})
    if(IE_SAMPLES_HEADERS)
        source_group("include" FILES ${IE_SAMPLES_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_SAMPLE_NAME} ${IE_SAMPLE_SOURCES} ${IE_SAMPLES_HEADERS})

    if(WIN32)
        set_target_properties(${IE_SAMPLE_NAME} PROPERTIES COMPILE_PDB_NAME ${IE_SAMPLE_NAME})
    endif()

    if(IE_SAMPLE_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_SAMPLE_NAME} PRIVATE ${IE_SAMPLE_INCLUDE_DIRECTORIES})
    endif()
    target_include_directories(${IE_SAMPLE_NAME} PRIVATE "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include")

    if(IE_SAMPLE_OPENCV_DEPENDENCIES)
        target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${OpenCV_LIBRARIES})
    endif()

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE IE::inference_engine
                                                    IE::inference_engine_plugin_api
                                                    ${IE_SAMPLE_DEPENDENCIES} gflags)

    # create global target with all samples / demo apps
    if(NOT TARGET ie_samples)
        add_custom_target(ie_samples ALL)
    endif()
    add_dependencies(ie_samples ${IE_SAMPLE_NAME})

    if(COMMAND add_cpplint_target AND NOT IE_SAMPLE_EXCLUDE_CPPLINT)
        add_cpplint_target(${IE_SAMPLE_NAME}_cpplint FOR_TARGETS ${IE_SAMPLE_NAME})
    endif()
endmacro()

# collect all samples subdirectories
file(GLOB samples_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
# skip building of unnecessary subdirectories
list(REMOVE_ITEM samples_dirs common thirdparty)
add_samples_to_build(${samples_dirs})

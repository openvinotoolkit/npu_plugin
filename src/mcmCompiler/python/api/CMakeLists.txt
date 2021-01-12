cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)

cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include/huffman_encoding)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../meta/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../meta/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../thirdparty/flatbuffers/include)
set(CMAKE_SWIG_FLAGS "")

set_source_files_properties(composition_interface.i PROPERTIES CPLUSPLUS on)
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.8.0") 
    swig_add_library(composition_api LANGUAGE python SOURCES composition_interface.i)
else()
    swig_add_module(composition_api python composition_interface.i)
    message("Consider upgrading to a newer version of CMake (at least 3.8.0)")
endif()

swig_link_libraries(composition_api ${PYTHON_LIBRARIES} mcmCompiler)
set_target_properties(${SWIG_MODULE_composition_api_REAL_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/api")

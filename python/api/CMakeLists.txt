cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 11)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_SWIG_FLAGS "")

set_source_files_properties(composition_interface.i PROPERTIES CPLUSPLUS on)
swig_add_library(composition_api LANGUAGE python SOURCES composition_interface.i)
swig_link_libraries(composition_api ${PYTHON_LIBRARIES} cm)
set_target_properties(${SWIG_MODULE_composition_api_REAL_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/api")

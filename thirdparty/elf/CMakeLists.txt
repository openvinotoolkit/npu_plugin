# {% copyright %}

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(TARGET_NAME elf)

add_library(${TARGET_NAME} INTERFACE)
target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(example)

add_subdirectory(test)

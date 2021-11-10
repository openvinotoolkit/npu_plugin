cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(TARGET_NAME host_parsing)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS})

add_library(${TARGET_NAME} INTERFACE ${HEADERS})

target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/parsing_lib/inc)

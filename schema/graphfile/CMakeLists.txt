cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(flatc ${PROJECT_SOURCE_DIR}/contrib/flatbuffers/flatc)

file(GLOB fbs_files *.fbs)

add_custom_command(

    OUTPUT
        ${CMAKE_SOURCE_DIR}/meta/schema/graphfile

    COMMAND
        flatc -o ${CMAKE_SOURCE_DIR}/meta/schema/graphfile ${fbs_files} --cpp --gen-object-api
    VERBATIM
    DEPENDS flatc
)

add_custom_target(
    generate-graphfile-headers
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/schema/graphfile
)
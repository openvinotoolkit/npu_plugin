cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

set(flatc_path ${CMAKE_BINARY_DIR}/contrib/flatbuffers/flatc)

file(GLOB fbs_files graphfile/src/schema/*.fbs)

add_custom_command(

    OUTPUT
        ${CMAKE_SOURCE_DIR}/meta/schema/graphfile/graphfile_generated.h

    COMMAND
        ${flatc_path} -o ${CMAKE_SOURCE_DIR}/meta/schema/graphfile ${fbs_files} --cpp --gen-object-api --force-empty
    VERBATIM
)

add_custom_target(
    generate-graphfile-headers
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/schema/graphfile/graphfile_generated.h
)

add_dependencies(generate-graphfile-headers flatc)

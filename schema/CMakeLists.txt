cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)


file(GLOB fbs_files graphfile/src/schema/*.fbs)

add_custom_command(

    OUTPUT
        ${META_DIR_PATH}/schema/graphfile/graphfile_generated.h

    COMMAND
        flatc -o ${META_DIR_PATH}/schema/graphfile ${fbs_files} --cpp --gen-object-api --force-empty
    VERBATIM
)

add_custom_target(
    generate-graphfile-headers
    DEPENDS ${META_DIR_PATH}/schema/graphfile/graphfile_generated.h
)

add_dependencies(generate-graphfile-headers flatc)


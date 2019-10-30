if(CMAKE_CROSSCOMPILING AND NOT TARGET gen_composition_api)
    message(STATUS "Cross-compiling, assumes schema/ files already generated...")
    return()
endif()

file(GLOB fbs_files graphfile/src/schema/*.fbs)

add_custom_command(
    OUTPUT
        ${META_DIR_PATH}/schema/graphfile/graphfile_generated.h
    COMMAND
        flatc -o ${META_DIR_PATH}/schema/graphfile ${fbs_files} --cpp --gen-object-api --force-empty
    VERBATIM
)

add_custom_target(generate-graphfile-headers
    DEPENDS
        ${META_DIR_PATH}/schema/graphfile/graphfile_generated.h
)

if(CMAKE_CROSSCOMPILING AND NOT TARGET gen_composition_api)
    message(STATUS "Cross-compiling, assumes schema/ files already generated...")
    return()
endif()

file(GLOB fbs_files graphfile/src/schema/*.fbs)

set(output_dir "${META_DIR_PATH}/schema/graphfile")

set(output_files "")
foreach(fbs_file IN LISTS fbs_files)
    get_filename_component(file_name_we "${fbs_file}" NAME_WE)
    set(output_file "${output_dir}/${file_name_we}_generated.h")
    list(APPEND output_files ${output_file})
endforeach()

add_custom_command(
    OUTPUT
        ${output_files}
    COMMAND
        flatc -o ${output_dir} --cpp --gen-object-api --force-empty ${fbs_files}
    DEPENDS
        ${fbs_files}
        flatc
    COMMENT
        "[flatc] Generate graphfile headers"
    VERBATIM
)

add_custom_target(generate-graphfile-headers
    DEPENDS
        ${output_files}
    COMMENT
        "[flatc] Generate graphfile headers"
)

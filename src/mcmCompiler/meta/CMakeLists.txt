#
# mcmCompilerGenCompAPI
#

if(NOT CMAKE_CROSSCOMPILING)
    add_executable(mcmCompilerCompAPIGenApp gen_composition_api.cpp ${base} ${logger})

    target_include_directories(mcmCompilerCompAPIGenApp
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            "${PROJECT_SOURCE_DIR}/include"
            "${META_DIR_PATH}/include"
            "${PROJECT_SOURCE_DIR}/include/huffman_encoding"
    )

    target_link_libraries(mcmCompilerCompAPIGenApp
        PRIVATE
            flatbuffers
    )

    file(MAKE_DIRECTORY "${META_DIR_PATH}/include/mcm" "${META_DIR_PATH}/src")

    add_custom_command(
        OUTPUT
            "${META_DIR_PATH}/include/mcm/compositional_model.hpp"
            "${META_DIR_PATH}/include/mcm/op_model.hpp"
            "${META_DIR_PATH}/src/compositional_model.cpp"
            "${META_DIR_PATH}/src/op_model.cpp"
        COMMAND
            mcmCompilerCompAPIGenApp "${META_DIR_PATH}"
        DEPENDS
            mcmCompilerCompAPIGenApp
        COMMENT
            "Generate compositional API"
        VERBATIM
    )

    add_custom_target(mcmCompilerGenCompAPI
        DEPENDS
            "${META_DIR_PATH}/include/mcm/compositional_model.hpp"
            "${META_DIR_PATH}/include/mcm/op_model.hpp"
            "${META_DIR_PATH}/src/compositional_model.cpp"
            "${META_DIR_PATH}/src/op_model.cpp"
        COMMENT
            "Generate compositional API"
    )
endif()

#
# mcmCompilerGenSchema
#

if(NOT CMAKE_CROSSCOMPILING)
    set(schema_src_dir "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema")
    file(GLOB schema_src_files "${schema_src_dir}/*.fbs")

    set(schema_dst_dir "${META_DIR_PATH}/include/schema/graphfile")
    file(MAKE_DIRECTORY ${schema_dst_dir})

    set(schema_dst_files)
    foreach(src_file IN LISTS schema_src_files)
        get_filename_component(file_name_we "${src_file}" NAME_WE)
        set(dst_file "${schema_dst_dir}/${file_name_we}_generated.h")
        list(APPEND schema_dst_files ${dst_file})
    endforeach()

    add_custom_command(
        OUTPUT
            ${schema_dst_files}
        COMMAND
            flatc -o ${schema_dst_dir} --cpp --gen-object-api --force-empty ${schema_src_files}
        DEPENDS
            ${schema_src_files}
            flatc
        COMMENT
            "[flatc] Generate graphfile headers"
        VERBATIM
    )

    add_custom_target(mcmCompilerGenSchema
        DEPENDS
            ${schema_dst_files}
        COMMENT
            "[flatc] Generate graphfile headers"
    )
endif()

#
# mcmCompilerGenCompAPI
#

if(NOT CMAKE_CROSSCOMPILING)
    add_executable(mcmCompilerCompAPIGenApp gen_composition_api.cpp ${base} ${logger})

    target_include_directories(mcmCompilerCompAPIGenApp
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/include/huffman_encoding"
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

if(NOT CMAKE_CROSSCOMPILING AND DEFINED MCM_GRAPH_SCHEMA_TAG)
    include(FetchContent)

    FetchContent_Declare(schema
         GIT_REPOSITORY "git@github.com:movidius/graphFile-schema.git"
         GIT_TAG ${MCM_GRAPH_SCHEMA_TAG})

    FetchContent_GetProperties(schema)

    if(NOT schema_POPULATED)
        FetchContent_Populate(schema)

        file(MAKE_DIRECTORY "${META_DIR_PATH}/include/schema")

        file(GLOB fbs_input_files "${schema_SOURCE_DIR}/src/schema/*.fbs")

        set(fbs_output_files)
        foreach(fbs_file IN LISTS fbs_input_files)
            get_filename_component(file_name_we "${fbs_file}" NAME_WE)
            set(output_file "${META_DIR_PATH}/include/schema/graphfile/${file_name_we}_generated.h")
            list(APPEND fbs_output_files ${output_file})
        endforeach()

        add_custom_command(
            OUTPUT
                ${fbs_output_files}
            COMMAND
                flatc -o "${META_DIR_PATH}/include/schema/graphfile" --cpp --gen-object-api --force-empty ${fbs_input_files}
            DEPENDS
                ${fbs_input_files}
                flatc
            COMMENT
                "[flatc] Generate graphfile headers"
            VERBATIM
        )

        add_custom_target(mcmCompilerGenSchema
            DEPENDS
                ${fbs_output_files}
            COMMENT
                "[flatc] Generate graphfile headers"
        )
    endif()
endif()

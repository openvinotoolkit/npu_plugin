#
# mcmCompilerGenSchema
#

vpux_add_flatc_target(mcmCompilerGenSchema
    SRC_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/graphFile-schema/src/schema"
    DST_DIR "${META_DIR_PATH}/include/schema/graphfile"
    ARGS "--gen-object-api" "--force-empty"
)
set_target_properties(mcmCompilerGenSchema PROPERTIES FOLDER "src")

#
# mcmCompilerGenCompAPI
#

add_executable(mcmCompilerCompAPIGenApp gen_composition_api.cpp ${base} ${logger})

add_dependencies(mcmCompilerCompAPIGenApp mcmCompilerGenSchema)

target_include_directories(mcmCompilerCompAPIGenApp
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        "${PROJECT_SOURCE_DIR}/include"
        "${META_DIR_PATH}"
        "${META_DIR_PATH}/include"
        "${PROJECT_SOURCE_DIR}/include/huffman_encoding"
        "${PROJECT_SOURCE_DIR}/../../thirdparty/bitcompactor/include"
)

target_link_libraries(mcmCompilerCompAPIGenApp
    PRIVATE
        flatbuffers
)

vpux_add_native_tool(mcmCompilerCompAPIGenApp
    ${PROJECT_SOURCE_DIR}
    EXEDIR "meta"
)

file(MAKE_DIRECTORY "${META_DIR_PATH}/include/mcm" "${META_DIR_PATH}/src")

add_custom_command(
    OUTPUT
        "${META_DIR_PATH}/include/mcm/compositional_model.hpp"
        "${META_DIR_PATH}/include/mcm/op_model.hpp"
        "${META_DIR_PATH}/src/compositional_model.cpp"
        "${META_DIR_PATH}/src/op_model.cpp"
    COMMAND
        ${mcmCompilerCompAPIGenApp_COMMAND} ${META_DIR_PATH}
    DEPENDS
        ${mcmCompilerCompAPIGenApp_COMMAND}
        ${mcmCompilerCompAPIGenApp_TARGET}
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
        ${mcmCompilerCompAPIGenApp_TARGET}
    COMMENT
        "Generate compositional API"
)

if (UNIX)
    set_source_files_properties(
        "${PROJECT_SOURCE_DIR}/../../thirdparty/bitcompactor/src/bitCompactor.cpp"
        PROPERTIES
            COMPILE_FLAGS "-Wno-unused-parameter"
    )
endif()

if(CMAKE_CROSSCOMPILING AND NOT TARGET gen_composition_api)
    # Old behavior.
    message(STATUS "Cross-compiling, assumes meta/ files already generated...")
    add_library(metamodel SHARED
        ${META_DIR_PATH}/src/compositional_model.cpp
        ${META_DIR_PATH}/src/op_model.cpp
    )
    target_link_libraries(metamodel model)
    return()
endif()

# Only build the generator if not crosscompiling.
if(NOT CMAKE_CROSSCOMPILING)
    message(STATUS "Native build, compile generator...")
    add_executable(gen_composition_api gen_composition_api.cpp ${base} ${logger})
endif()

file(MAKE_DIRECTORY ${META_DIR_PATH}/include/mcm ${META_DIR_PATH}/src)

add_custom_command(
    OUTPUT
        ${META_DIR_PATH}/include/mcm/compositional_model.hpp
        ${META_DIR_PATH}/include/mcm/op_model.hpp
        ${META_DIR_PATH}/src/compositional_model.cpp
        ${META_DIR_PATH}/src/op_model.cpp
    COMMAND
        gen_composition_api ${META_DIR_PATH}
    DEPENDS
        gen_composition_api
    COMMENT
        "Generate compositional API"
    VERBATIM
)

add_custom_target(generate-comp-api
    DEPENDS
        ${META_DIR_PATH}/include/mcm/compositional_model.hpp
        ${META_DIR_PATH}/include/mcm/op_model.hpp
        ${META_DIR_PATH}/src/compositional_model.cpp
        ${META_DIR_PATH}/src/op_model.cpp
    COMMENT
        "Generate compositional API"
)

if (MSVC)
    add_library(metamodel OBJECT
        ${META_DIR_PATH}/src/compositional_model.cpp
        ${META_DIR_PATH}/src/op_model.cpp
        $<TARGET_OBJECTS:model>
    )
else()
    add_library(metamodel SHARED
        ${META_DIR_PATH}/src/compositional_model.cpp
        ${META_DIR_PATH}/src/op_model.cpp
    )

    target_link_libraries(metamodel model)
    if(${PROFILER})
        target_link_libraries(metamodel easy_profiler)
    endif()
endif()

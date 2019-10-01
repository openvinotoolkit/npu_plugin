cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

add_executable(gen_composition_api gen_composition_api.cpp ${base} ${logger})

file(MAKE_DIRECTORY ${META_DIR_PATH}/include/mcm ${META_DIR_PATH}/src)

add_custom_command(
    OUTPUT
        ${META_DIR_PATH}/include/mcm/compositional_model.hpp
        ${META_DIR_PATH}/include/mcm/op_model.hpp
        ${META_DIR_PATH}/src/compositional_model.cpp
        ${META_DIR_PATH}/src/op_model.cpp
       
		COMMAND
			$<TARGET_FILE:gen_composition_api>
		VERBATIM
		DEPENDS $<TARGET_FILE:gen_composition_api>
)

add_custom_target(
    generate-comp-api
    DEPENDS ${META_DIR_PATH}/include/mcm/compositional_model.hpp
    DEPENDS ${META_DIR_PATH}/include/mcm/op_model.hpp
    DEPENDS ${META_DIR_PATH}/src/compositional_model.cpp
    DEPENDS ${META_DIR_PATH}/src/op_model.cpp
)


if (MSVC)
	add_library(metamodel OBJECT
		${META_DIR_PATH}/src/compositional_model.cpp
		${META_DIR_PATH}/src/op_model.cpp
		$<TARGET_OBJECTS:model>
		$<TARGET_OBJECTS:corebase>
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
endif(MSVC)
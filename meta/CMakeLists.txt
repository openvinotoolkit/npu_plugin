cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

add_executable(gen_composition_api gen_composition_api.cpp ${base} ${logger})

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/meta/include/mcm ${CMAKE_SOURCE_DIR}/meta/src/)

add_custom_command(
    OUTPUT
        ${CMAKE_SOURCE_DIR}/meta/include/mcm/compositional_model.hpp
        ${CMAKE_SOURCE_DIR}/meta/include/mcm/op_model.hpp
        ${CMAKE_SOURCE_DIR}/meta/include/mcm/recorded_compositional_model.hpp
        ${CMAKE_SOURCE_DIR}/meta/src/compositional_model.cpp
        ${CMAKE_SOURCE_DIR}/meta/src/op_model.cpp
        ${CMAKE_SOURCE_DIR}/meta/src/recorded_compositional_model.cpp
       
		COMMAND
			$<TARGET_FILE:gen_composition_api>
		VERBATIM
		DEPENDS $<TARGET_FILE:gen_composition_api>
)

add_custom_target(
    generate-comp-api
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/include/mcm/compositional_model.hpp
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/include/mcm/op_model.hpp
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/include/mcm/recorded_compositional_model.hpp
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/src/compositional_model.cpp
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/src/op_model.cpp
    DEPENDS ${CMAKE_SOURCE_DIR}/meta/src/recorded_compositional_model.cpp
)


if (MSVC)
	add_library(metamodel OBJECT
		${CMAKE_SOURCE_DIR}/meta/src/compositional_model.cpp
		${CMAKE_SOURCE_DIR}/meta/src/op_model.cpp
		${CMAKE_SOURCE_DIR}/meta/src/recorded_compositional_model.cpp
		$<TARGET_OBJECTS:model>
		$<TARGET_OBJECTS:corebase>
	)
else()
	add_library(metamodel SHARED
		${CMAKE_SOURCE_DIR}/meta/src/compositional_model.cpp
		${CMAKE_SOURCE_DIR}/meta/src/op_model.cpp
		${CMAKE_SOURCE_DIR}/meta/src/recorded_compositional_model.cpp
	)
	target_link_libraries(metamodel model)

endif(MSVC)
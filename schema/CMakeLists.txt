cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

if (FLATBUFFERS_FOUND)
	set(flatc_path ${FLATBUFFERS_COMPILER})
else()
   	set(flatc_path ${CMAKE_BINARY_DIR}/contrib/flatbuffers/flatc)
endif(FLATBUFFERS_FOUND)



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

if (FLATBUFFERS_NOT_FOUND)
   add_dependencies(generate-graphfile-headers flatc)
endif(FLATBUFFERS_NOT_FOUND)


if(NOT DEFINED "ENV{VPU_CLC_PATH}")
    message(WARNING "VPU_CLC_PATH variable for OpenCL compiler is not set.")
elseif(NOT EXISTS "$ENV{VPU_CLC_PATH}/bin/clc")
    message(WARNING "Can't find 'clc' file in '\${VPU_CLC_PATH}/bin' folder:\n$ENV{VPU_CLC_PATH}/bin")
else()
    set(RUN_CUSTOM_LAYER_TESTS 1)

    set(VPU_CLC_PATH $ENV{VPU_CLC_PATH})
    set(KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
    set(ELF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/elf")

    file(GLOB CL_FILES "${KERNELS_DIR}/*.cl")

    file(MAKE_DIRECTORY "${ELF_DIR}")

    foreach(cl_file IN LISTS CL_FILES)
        get_filename_component(cl_file_name ${cl_file} NAME_WE)

        set(out_file "${ELF_DIR}/${cl_file_name}.elf")
        list(APPEND CUSTOM_KERNELS ${out_file})

        add_custom_command(
            OUTPUT ${out_file}
            COMMAND
                ${CMAKE_COMMAND} -E env
                    "SHAVE_LDSCRIPT_DIR=${VPU_CLC_PATH}/ldscripts-kmb/"
                    "SHAVE_MA2X8XLIBS_DIR=${VPU_CLC_PATH}/lib"
                    "SHAVE_MOVIASM_DIR=${VPU_CLC_PATH}/bin"
                    "SHAVE_MYRIAD_LD_DIR=${VPU_CLC_PATH}/bin"
                ${VPU_CLC_PATH}/bin/clc --strip-binary-header ${cl_file} -o ${out_file}
            MAIN_DEPENDENCY ${cl_file}
            COMMENT "[CLC] Compiling ${cl_file_name}.elf"
            VERBATIM)
    endforeach()
endif()
add_custom_target(compile_custom_kernels
        DEPENDS ${CUSTOM_KERNELS}
        COMMENT "[CLC] Compiled custom kernels")

file(GLOB_RECURSE NATIVE_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
list(FILTER NATIVE_TEST_FILES EXCLUDE REGEX ".*custom_.*.cpp")
set(TEST_FILES ${NATIVE_TEST_FILES})

if (RUN_CUSTOM_LAYER_TESTS)
    file(GLOB_RECURSE CUSTOM_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
    list(FILTER CUSTOM_TEST_FILES INCLUDE REGEX ".*custom_.*.cpp")
    list(APPEND TEST_FILES ${CUSTOM_TEST_FILES})
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common/")

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(target_name ${test_file} NAME_WE)

    add_executable(${target_name} ${test_file})
    target_link_libraries(${target_name} cm)
    set(mcm_blob "${CMAKE_CURRENT_BINARY_DIR}/output/mcm.blob")
    set(out_blob "${CMAKE_CURRENT_SOURCE_DIR}/blobs/${target_name}.blob")
    set(blobs_info "${CMAKE_CURRENT_SOURCE_DIR}/blobs_info")
    set(out_blob_info "${blobs_info}/${target_name}.csv")

    add_custom_command(
        OUTPUT ${out_blob} ${out_blob_info}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${blobs_info}
        COMMAND ${target_name} > ${out_blob_info}
        COMMAND ${CMAKE_COMMAND} -E copy ${mcm_blob} ${out_blob}
        DEPENDS compile_custom_kernels
        COMMENT "Compiling ${target_name}.blob"
        VERBATIM)

    list(APPEND all_blobs ${out_blob})
endforeach()

add_custom_target(gen_blobs
        DEPENDS ${all_blobs}
        COMMENT "Compiled test blobs")

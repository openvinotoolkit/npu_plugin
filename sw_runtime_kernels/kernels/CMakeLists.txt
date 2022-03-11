#
# Copyright 2022 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

cmake_minimum_required(VERSION 3.15)

project(kernels)

#set(BUILD_BLOB_BINARIES OFF CACHE BOOL "" FORCE)
#set(BUILD_JTAG_BINARIES OFF CACHE BOOL "" FORCE)

message(STATUS "CUSTOM_KERNELS_DIR = ${CUSTOM_KERNELS_DIR}")
message(STATUS "CUSTOM_KERNELS_LIST = ${CUSTOM_KERNELS_LIST}")
message(STATUS "BUILD_STD_KERNELS = ${BUILD_STD_KERNELS}")
message(STATUS "BUILD_PSS_KERNELS = ${BUILD_PSS_KERNELS}")
message(STATUS "BUILD_BLOB_BINARIES = ${BUILD_BLOB_BINARIES}")
message(STATUS "BUILD_JTAG_BINARIES = ${BUILD_JTAG_BINARIES}")

#message(STATUS "source files: ${kernel_srcs_list}")

if (NOT DEFINED ENV{MV_TOOLS_DIR})
  message(FATAL_ERROR "MV_TOOLS_DIR environment variable must be defined")
endif()

if (DEFINED ENV{MV_TOOLS_VERSION})
  set(MV_TOOLS_VERSION "$ENV{MV_TOOLS_VERSION}")
else()
  execute_process(
    COMMAND grep "mv_tools_version" ${CMAKE_SOURCE_DIR}/../firmware_vpu_revision.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE tools_version_string
  )
  string(REPLACE "mv_tools_version:	" "" tools_version_string "${tools_version_string}")
  string(STRIP "${tools_version_string}" tools_version_string)
  set(MV_TOOLS_VERSION "${tools_version_string}")
endif()

if (NOT DEFINED ENV{FIRMWARE_VPU_DIR})
  message(FATAL_ERROR "FIRMWARE_VPU_DIR environment variable must be defined")
endif()

set(MV_TOOLS_DIR "$ENV{MV_TOOLS_DIR}")
set(FIRMWARE_VPU_DIR "$ENV{FIRMWARE_VPU_DIR}")

message(STATUS "MV_TOOLS_DIR     = ${MV_TOOLS_DIR}")
message(STATUS "MV_TOOLS_VERSION = ${MV_TOOLS_VERSION}")
message(STATUS "FIRMWARE_VPU_DIR = ${FIRMWARE_VPU_DIR}")

message(STATUS "source directory: ${CMAKE_SOURCE_DIR}")

set(kernel_src_dir_default "src")
set(kernel_cpunum_default "3010")
set(optimization_opts_default "-O3")

set(include_dirs_list_default
  "${CMAKE_SOURCE_DIR}/inc"
  "${CMAKE_SOURCE_DIR}/inc/3720"
  "${CMAKE_SOURCE_DIR}/common/inc"
  "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc"
)

set(define_symbols_list_default
  "__shave_nn__"
  "CONFIG_TARGET_SOC_3720"
)

set(always_inline_default "yes")

set(mv_tools_compile "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile")
set(mv_tools_link "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld")
set(mv_tools_objcopy "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy")
set(host_tools_xxd "/usr/bin/xxd")

if (BUILD_JTAG_BINARIES AND (NOT EXISTS "${host_tools_xxd}"))
  message(FATAL_ERROR "xxd utility is required for JTAG tests but is not installed")
endif()

set(target_prefix "sk.")
set(obj_suffix ".o")
set(elf_suffix ".elf")
set(blob_text_suffix ".text")
set(blob_data_suffix ".data")
set(jtag_text_suffix ".text.xdat")
set(jtag_data_suffix ".data.xdat")

set(link_script_file "${CMAKE_SOURCE_DIR}/prebuild/shave_kernel.ld")

set(link_libraries_list
  "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibm.a"
  "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcxx.a"
  "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibc.a"
  "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/30xxxx-leon/mlibcrt.a"
)

set(kernel_descrip_dir "${CUSTOM_KERNELS_DIR}")
if (kernel_descrip_dir STREQUAL "")
  set(kernel_descrip_dir "${CMAKE_SOURCE_DIR}/descrip/")
endif()

set(kernel_descrip_list "${CUSTOM_KERNELS_LIST}")
if (NOT kernel_descrip_list STREQUAL "")
  message(STATUS "[NOT CUSTOM_KERNELS_LIST STREQUAL \"\"] => T")
  set(kernel_descrip_list ${CUSTOM_KERNELS_LIST})
  list(TRANSFORM kernel_descrip_list PREPEND "${kernel_descrip_dir}")
  message(STATUS "kernel_descrip_list = ${kernel_descrip_list}")
else()
  message(STATUS "[NOT CUSTOM_KERNELS_LIST STREQUAL \"\"] => F")
  set(kernel_descrip_list "")
  if (BUILD_STD_KERNELS)
    file(GLOB std_list "${kernel_descrip_dir}*.txt")
    list(APPEND kernel_descrip_list ${std_list})
  endif()
  if (BUILD_PSS_KERNELS)
    file(GLOB pss_list "${kernel_descrip_dir}pss/*.txt")
    list(APPEND kernel_descrip_list ${pss_list})
  endif()
endif()

message(STATUS "kernel_descrip_list = ${kernel_descrip_list}")

foreach(kernel_descrip ${kernel_descrip_list})

  set(kernel_entry "")
  set(kernel_src "<unknown>")
  set(kernel_src_dir "${kernel_src_dir_default}")
  set(kernel_cpunum "${kernel_cpunum_default}")
  set(optimization_opts "${optimization_opts_default}")
  set(include_dirs_list "")
  set(define_symbols_list "")
  set(always_inline "${always_inline_default}")
  set(extra_src_list "")

  include("${kernel_descrip}")
  get_filename_component(kernel_name ${kernel_src} NAME_WE)

  if (kernel_entry STREQUAL "")
    set(kernel_entry "${kernel_name}")
  endif()

  message(STATUS "kernel_cpunum = ${kernel_cpunum}")

  if (kernel_cpunum STREQUAL "3720")
    list(APPEND define_symbols_list "USE_3720_INTSTRUCTIONS")
  endif()

  if (always_inline STREQUAL "yes")
    list(APPEND define_symbols_list "CONFIG_ALWAYS_INLINE")
  endif()

  list(PREPEND include_dirs_list ${include_dirs_list_default})
  list(PREPEND define_symbols_list ${define_symbols_list_default})

  set(kernel_cpu "${kernel_cpunum}xx")
  set(kernel_cpu_suffix ".${kernel_cpunum}xx")

  set(kernel_src_file "${kernel_src_dir}/${kernel_src}")
  set(kernel_src_path "${CMAKE_SOURCE_DIR}/${kernel_src_file}")

  message(STATUS "kernel = ${kernel_name} ${kernel_src_file}")

  list(TRANSFORM include_dirs_list PREPEND "-I")
  list(TRANSFORM define_symbols_list PREPEND "-D")

  message(STATUS "define_symbols_list = ${define_symbols_list}")
  message(STATUS "include_dirs_list = ${include_dirs_list}")

  if (BUILD_BLOB_BINARIES OR BUILD_JTAG_BINARIES)
    set(obj_file "${kernel_src}${kernel_cpu_suffix}${obj_suffix}")
    set(obj_path "${CMAKE_BINARY_DIR}/${obj_file}")
    message(STATUS "obj = ${obj_path}")

    set(elf_file "${kernel_name}${kernel_cpu_suffix}${elf_suffix}")
    set(elf_path "${CMAKE_BINARY_DIR}/${elf_file}")
    message(STATUS "elf = ${elf_path}")

    add_custom_target("${obj_file}"
      DEPENDS "${kernel_src_file}"
      COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${include_dirs_list} ${define_symbols_list} -c "${kernel_src_path}" -o "${obj_file}"
    )
    set(obj_file_list "${obj_file}")

    if (NOT extra_src_list STREQUAL "")
      foreach(extra_src_file ${extra_src_list})
        message(STATUS "extra_src_file = ${extra_src_file}")
        get_filename_component(src_name ${extra_src_file} NAME_WE)
        set(obj_file "${src_name}${obj_suffix}")
        add_custom_target("${obj_file}"
          DEPENDS "${extra_src_file}"
          COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${include_dirs_list} ${define_symbols_list} -c "${extra_src_file}" -o "${obj_file}"
        )
        list(APPEND obj_file_list "${obj_file}")
      endforeach()
    endif()

    add_custom_target("${elf_file}"
      DEPENDS ${obj_file_list}
      COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list} -EL ${link_libraries_list} --output "${elf_file}"
    )
  endif()

  if (BUILD_BLOB_BINARIES)
    set(blob_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${blob_text_suffix}")
    set(blob_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${blob_data_suffix}")
    set(blob_text_path "${CMAKE_BINARY_DIR}/${blob_text_file}")
    set(blob_data_path "${CMAKE_BINARY_DIR}/${blob_data_file}")
    message(STATUS "blob_text = ${blob_text_path}")
    message(STATUS "blob_data = ${blob_data_path}")

    add_custom_target("${blob_text_file}" ALL
      COMMAND "${mv_tools_objcopy}" -O binary --only-section=.text "${elf_file}" "${blob_text_file}"
      DEPENDS "${elf_file}"
    )
    add_custom_target("${blob_data_file}" ALL
      COMMAND "${mv_tools_objcopy}" -O binary --only-section=.data "${elf_file}" "${blob_data_file}"
      DEPENDS "${elf_file}"
    )
  endif()

  if (BUILD_JTAG_BINARIES)
    set(jtag_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${jtag_text_suffix}")
    set(jtag_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${jtag_data_suffix}")
    set(jtag_text_path "${CMAKE_BINARY_DIR}/${jtag_text_file}")
    set(jtag_data_path "${CMAKE_BINARY_DIR}/${jtag_data_file}")
    message(STATUS "jtag_text = ${jtag_text_path}")
    message(STATUS "jtag_data = ${jtag_data_path}")

    add_custom_target("${jtag_text_file}" ALL
      COMMAND ${host_tools_xxd} -i "${blob_text_file}" "${jtag_text_path}"
      DEPENDS "${blob_text_file}"
    )
    add_custom_target("${jtag_data_file}" ALL
      COMMAND ${host_tools_xxd} -i "${blob_data_file}" "${jtag_data_path}"
      DEPENDS "${blob_data_file}"
    )
  endif()
endforeach()

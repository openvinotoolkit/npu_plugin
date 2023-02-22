# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

cmake_minimum_required(VERSION 3.13.4)

project(kernels)

if(DEFINED TARGET_BINARY_DIR)
  set(target_binary_dir "${TARGET_BINARY_DIR}")
else()
  set(target_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin")
endif()

if(DEFINED TARGET_JTAG_DIR)
  set(target_jtag_dir "${TARGET_JTAG_DIR}")
else()
  set(target_jtag_dir "${target_binary_dir}/..")
endif()

set(kernel_src_dir_default "src")

set(include_dirs_list_default
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc/3720"
)

set(define_symbols_list_default
  "__act_shave__"
  "CONFIG_TARGET_SOC_3720"
)

set(always_inline_default "no")
set(rt_kernel_default "no")

set(target_prefix "sk.")
set(obj_suffix ".o")
set(elf_suffix ".elf")
set(blob_text_suffix ".text")
set(blob_data_suffix ".data")
set(jtag_text_suffix ".text.xdat")
set(jtag_data_suffix ".data.xdat")

set(link_script_file_default "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_kernel.ld")

set(kernel_descrip_dir "${CUSTOM_KERNELS_DIR}")

if(kernel_descrip_dir STREQUAL "")
  set(kernel_descrip_dir "${CMAKE_CURRENT_SOURCE_DIR}/descrip/")
endif()

option(ENABLE_SHAVE_BINARIES_BUILD "Enable shave binaries build, if disabled, prebuilt binaries will be used" OFF)
option(ENABLE_MANAGEMENT_KERNEL_BUILD "Enable management kernel build" OFF)
option(ENABLE_PSS_BINARIES_BUILD "Enable pss binaries build" ON)
option(ENABLE_JTAG_BINARIES_BUILD "Enable jtag binaries build" OFF)

set(use_prebuilt_binaries OFF)

if(NOT ENABLE_SHAVE_BINARIES_BUILD)
  set(use_prebuilt_binaries ON)

  if(ENABLE_MANAGEMENT_KERNEL_BUILD)
    message(FATAL_ERROR "When ENABLE_MANAGEMENT_KERNEL_BUILD=ON please use ENABLE_SHAVE_BINARIES_BUILD=`ON`")
  endif()
endif()

if(DEFINED ENV{MV_TOOLS_DIR} AND DEFINED ENV{MV_TOOLS_VERSION})
  set(MV_TOOLS_DIR "$ENV{MV_TOOLS_DIR}")
  set(MV_TOOLS_VERSION "$ENV{MV_TOOLS_VERSION}")

  if(UNIX)
    set(mv_tools_compile "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile")
    set(mv_tools_link "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld")
    set(mv_tools_objcopy "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy")
  elseif(WIN32)
    set(mv_tools_compile "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/win32/bin/moviCompile.exe")
    set(mv_tools_link "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/win32/sparc-myriad-rtems-9.2.0/bin/sparc-myriad-rtems-ld.exe")
    set(mv_tools_objcopy "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/win32/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy.exe")
  endif()

  set(link_libraries_list_VPU3720
    "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/37xxxx/mlibc_lgpl.a"
    "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/37xxxx/mlibm.a"
    "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/37xxxx/mlibcxx.a"
    "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/37xxxx/mlibc.a"
    "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/common/moviCompile/lib/37xxxx/mlibcrt.a"
  )
else()
  if(ENABLE_SHAVE_BINARIES_BUILD)
    message(FATAL_ERROR "MV_TOOLS_DIR and MV_TOOLS_VERSION environment variables must be defined when ENABLE_SHAVE_BINARIES_BUILD=ON")
  endif()

  if(ENABLE_MANAGEMENT_KERNEL_BUILD)
    message(FATAL_ERROR "MV_TOOLS_DIR and MV_TOOLS_VERSION environment variables must be defined when BUILD_MANGEMENT_KERNEL=ON")
  endif()
endif()

if(ENABLE_MANAGEMENT_KERNEL_BUILD)
  if(NOT DEFINED ENV{FIRMWARE_VPU_DIR})
    message(FATAL_ERROR "FIRMWARE_VPU_DIR environment variable must be defined when ENABLE_MANAGEMENT_KERNEL_BUILD=ON")
  endif()

  set(FIRMWARE_VPU_DIR "$ENV{FIRMWARE_VPU_DIR}")
  list(APPEND include_dirs_list_default "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc")
endif()

if(ENABLE_JTAG_BINARIES_BUILD)
  set(host_tools_xxd "/usr/bin/xxd")

  if(NOT EXISTS "${host_tools_xxd}")
    message(FATAL_ERROR "xxd utility is required when ENABLE_JTAG_BINARIES_BUILD=ON")
  endif()
endif()

add_custom_target(act_shave_kernels_ready)
add_custom_target(act_shave_kernels_build)
add_custom_target(act_shave_jtag_kernels_build)

add_custom_target(act_shave_kernels_copy
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin" "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin"
)

add_custom_target(update_shave_prebuilt_binaries
  DEPENDS ${act_shave_kernels}
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${target_binary_dir}/" "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/"
)

add_dependencies(act_shave_kernels_ready act_shave_kernels_copy)

set(kernel_descrip_list "${CUSTOM_KERNELS_LIST}")

if(NOT kernel_descrip_list STREQUAL "")
  set(kernel_descrip_list ${CUSTOM_KERNELS_LIST})
  list(TRANSFORM kernel_descrip_list PREPEND "${kernel_descrip_dir}")
else()
  set(kernel_descrip_list "")

  file(GLOB kernel_list "${kernel_descrip_dir}*.txt")
  list(APPEND kernel_descrip_list ${kernel_list})

  if(ENABLE_PSS_BINARIES_BUILD)
    file(GLOB pss_list "${kernel_descrip_dir}pss/*.txt")
    list(APPEND kernel_descrip_list ${pss_list})
  endif()
endif()

set(act_shave_kernels "")
set(compiled_extra_list "")
file(MAKE_DIRECTORY ${target_binary_dir})

foreach(kernel_descrip ${kernel_descrip_list})
  set(kernel_entry "")
  set(kernel_src "<unknown>")
  set(kernel_src_dir "${kernel_src_dir_default}")
  set(kernel_cpunum "3720")
  set(optimization_opts "-O3")
  set(include_dirs_list "")
  set(define_symbols_list "")
  set(always_inline "${always_inline_default}")
  set(rt_kernel "${rt_kernel_default}")
  set(extra_src_list "")
  set(link_script_file "${link_script_file_default}")

  get_filename_component(kernel_descrip_path ${kernel_descrip} DIRECTORY)

  include("${kernel_descrip}")
  get_filename_component(kernel_name ${kernel_src} NAME_WE)

  if(kernel_entry STREQUAL "")
    set(kernel_entry "${kernel_name}")
  endif()

  if(kernel_cpunum STREQUAL "3720")
    list(APPEND define_symbols_list "USE_3720_INTSTRUCTIONS")
    set(link_libraries_list ${link_libraries_list_VPU3720})
  endif()

  if(always_inline STREQUAL "yes")
    list(APPEND define_symbols_list "CONFIG_ALWAYS_INLINE")
  endif()

  list(INSERT include_dirs_list 0 ${include_dirs_list_default})
  list(INSERT define_symbols_list 0 ${define_symbols_list_default})

  set(kernel_cpu "${kernel_cpunum}xx")
  set(kernel_cpu_suffix ".${kernel_cpunum}xx")

  set(kernel_src_file "${kernel_src_dir}/${kernel_src}")

  set(blob_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${blob_text_suffix}")
  set(blob_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${blob_data_suffix}")
  set(jtag_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${jtag_text_suffix}")
  set(jtag_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${jtag_data_suffix}")

  if(rt_kernel STREQUAL "yes")
    if(NOT ENABLE_MANAGEMENT_KERNEL_BUILD)
      add_custom_target("${kernel_name}${kernel_cpu_suffix}_kernel_copy"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${target_binary_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${blob_text_file}" "${target_binary_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${blob_data_file}" "${target_binary_dir}"
      )
      add_dependencies(act_shave_kernels_build "${kernel_name}${kernel_cpu_suffix}_kernel_copy")

      if(ENABLE_JTAG_BINARIES_BUILD)
        add_custom_target("${kernel_name}${kernel_cpu_suffix}_jtag_kernel_copy"
          COMMAND ${CMAKE_COMMAND} -E make_directory "${target_jtag_dir}"
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/${jtag_text_file}" "${target_jtag_dir}"
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/${jtag_data_file}" "${target_jtag_dir}"
        )
        add_dependencies(act_shave_jtag_kernels_build "${kernel_name}${kernel_cpu_suffix}_jtag_kernel_copy")
      endif()

      continue()
    endif()

    set(kernel_src_path "${kernel_src_file}")
  else()
    set(kernel_src_path "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_src_file}")
  endif()

  list(APPEND act_shave_kernels ${kernel_src_file})

  list(TRANSFORM include_dirs_list PREPEND "-I")
  list(TRANSFORM define_symbols_list PREPEND "-D")

  set(obj_file "${kernel_src}${kernel_cpu_suffix}${obj_suffix}")

  set(elf_file "${kernel_name}${kernel_cpu_suffix}${elf_suffix}")
  set(elf_path "${target_binary_dir}/${elf_file}")

  add_custom_command(
    OUTPUT "${obj_file}"
    DEPENDS "${kernel_src_file}"
    COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${include_dirs_list} ${define_symbols_list} -c "${kernel_src_path}" -o "${obj_file}"
  )
  set(obj_file_list "${obj_file}")

  if(NOT extra_src_list STREQUAL "")
    foreach(extra_src_file ${extra_src_list})
      get_filename_component(src_name ${extra_src_file} NAME_WE)
      set(obj_file "${src_name}${kernel_cpu}${obj_suffix}")

      # Compile once and check list to avoid double compilation
      if(NOT "${obj_file}" IN_LIST compiled_extra_list)
        add_custom_command(
          OUTPUT "${obj_file}"
          DEPENDS "${extra_src_file}"
          COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} -O3 ${include_dirs_list} ${define_symbols_list} -c "${extra_src_file}" -o "${obj_file}"
        )
        list(APPEND compiled_extra_list "${obj_file}")
      endif()

      # We still want to link the obj file even if it's already compiled
      list(APPEND obj_file_list "${obj_file}")
    endforeach()
  endif()

  add_custom_command(
    OUTPUT "${elf_path}"
    DEPENDS ${obj_file_list}
    COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list} -EL ${link_libraries_list} --output "${elf_path}"
  )

  set(blob_text_path "${target_binary_dir}/${blob_text_file}")
  set(blob_data_path "${target_binary_dir}/${blob_data_file}")

  add_custom_target("${blob_text_file}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${target_binary_dir}"
    COMMAND "${mv_tools_objcopy}" -O binary --only-section=.text "${elf_path}" "${blob_text_path}"
    DEPENDS "${elf_path}"
  )
  add_custom_target("${blob_data_file}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${target_binary_dir}"
    COMMAND "${mv_tools_objcopy}" -O binary --only-section=.data "${elf_path}" "${blob_data_path}"
    DEPENDS "${elf_path}"
  )

  add_dependencies(act_shave_kernels_build "${blob_text_file}" "${blob_data_file}")

  if(ENABLE_JTAG_BINARIES_BUILD)
    set(jtag_text_path "${CMAKE_CURRENT_BINARY_DIR}/${jtag_text_file}")
    set(jtag_data_path "${CMAKE_CURRENT_BINARY_DIR}/${jtag_data_file}")

    add_custom_target("${jtag_text_file}"
      COMMAND ${host_tools_xxd} -i "${blob_text_file}" "${jtag_text_path}"
      WORKING_DIRECTORY "${target_binary_dir}"
      DEPENDS "${blob_text_file}"
    )
    add_custom_target("${jtag_data_file}"
      COMMAND ${host_tools_xxd} -i "${blob_data_file}" "${jtag_data_path}"
      WORKING_DIRECTORY "${target_binary_dir}"
      DEPENDS "${blob_data_file}"
    )

    add_dependencies(act_shave_jtag_kernels_build "${jtag_text_file}" "${jtag_data_file}")
  endif()
endforeach()

# Add file-level and target-level dependencies
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp"
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/embed_shave_binaries.cmake"
  DEPENDS ${act_shave_kernels} act_shave_kernels_ready
)

add_library(act_shave_kernels_lib OBJECT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp")

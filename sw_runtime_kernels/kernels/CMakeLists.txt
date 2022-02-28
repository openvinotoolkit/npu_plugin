cmake_minimum_required(VERSION 2.6)

project(kernels)

#set(BUILD_BLOB_BINARIES OFF CACHE BOOL "" FORCE)
#set(BUILD_JTAG_BINARIES OFF CACHE BOOL "" FORCE)

message(STATUS "BUILD_BLOB_BINARIES = ${BUILD_BLOB_BINARIES}")
message(STATUS "BUILD_JTAG_BINARIES = ${BUILD_JTAG_BINARIES}")

set(kernel_srcs_list dummy.cpp elu_fp16.cpp exp_fp16.cpp hswish_fp16.cpp reorder_fp16.cpp sigmoid_fp16.c singleShaveInterpolate.cpp singleShaveMVN.cpp singleShaveSoftmax.cpp sqrt_fp16.cpp tanh_fp16.cpp nnActEntry.cpp)

set(kernel_src_dir_DEFAULT "src")
set(kernel_src_dir_nnActEntry "act_runtime/src")

set(kernel_cpunum_DEFAULT "3010")
set(kernel_cpunum_elu_fp16 "3270")
set(kernel_cpunum_exp_fp16 "3270")

message(STATUS "source files: ${kernel_srcs_list}")

if(NOT DEFINED ENV{MV_TOOLS_DIR})
  message(FATAL_ERROR "MV_TOOLS_DIR environment variable must be defined")
endif()

if(NOT DEFINED ENV{MV_TOOLS_VERSION})
  message(FATAL_ERROR "MV_TOOLS_VERSION environment variable must be defined")
endif()

if(NOT DEFINED ENV{FIRMWARE_VPU_DIR})
  message(FATAL_ERROR "FIRMWARE_VPU_DIR environment variable must be defined")
endif()

set(MV_TOOLS_DIR "$ENV{MV_TOOLS_DIR}")
set(MV_TOOLS_VERSION "$ENV{MV_TOOLS_VERSION}")
set(FIRMWARE_VPU_DIR "$ENV{FIRMWARE_VPU_DIR}")

message(STATUS "source directory: ${CMAKE_SOURCE_DIR}")

if(DEFINED ENV{KERNEL_DIR})
  set(KERNEL_DIR "$ENV{KERNEL_DIR}")
else()
  set(KERNEL_DIR "${CMAKE_SOURCE_DIR}")
  message(WARNING "KERNEL_DIR environment variable is not set; using default ${KERNEL_DIR}")
endif()

message(STATUS "MV_TOOLS_DIR     = ${MV_TOOLS_DIR}")
message(STATUS "MV_TOOLS_VERSION = ${MV_TOOLS_VERSION}")
message(STATUS "FIRMWARE_VPU_DIR = ${FIRMWARE_VPU_DIR}")
message(STATUS "KERNEL_DIR       = ${KERNEL_DIR}")

set(mv_tools_compile "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviCompile")
set(mv_tools_link "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld")
set(mv_tools_objcopy "${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-objcopy")
set(host_tools_xxd "/usr/bin/xxd")

if(BUILD_JTAG_BINARIES AND (NOT EXISTS "${host_tools_xxd}"))
  message(FATAL_ERROR "xxd utility is required for JTAG tests but is not installed")
endif()

set(blob_text_suffix ".text")
set(blob_data_suffix ".data")
set(jtag_text_suffix ".text.xdat")
set(jtag_data_suffix ".data.xdat")

foreach(kernel_src ${kernel_srcs_list})
  get_filename_component(kernel_name ${kernel_src} NAME_WE)

  set(kernel_src_dir ${kernel_src_dir_${kernel_name}})
  if ("${kernel_src_dir}" STREQUAL "")
    set(kernel_src_dir ${kernel_src_dir_DEFAULT})
  endif()

  set(kernel_cpunum ${kernel_cpunum_${kernel_name}})
  if ("${kernel_cpunum}" STREQUAL "")
    set(kernel_cpunum ${kernel_cpunum_DEFAULT})
  endif()
  set(kernel_cpu_suffix ".${kernel_cpunum}xx")

  set(kernel_src_file "${kernel_src_dir}/${kernel_src}")
  set(kernel_src_path "${KERNEL_DIR}/${kernel_src_file}")

  message(STATUS "kernel = ${kernel_name} ${kernel_src_file}")

  if(BUILD_BLOB_BINARIES)
    set(target_blob_text_file "${kernel_name}${kernel_cpu_suffix}${blob_text_suffix}")
    set(target_blob_data_file "${kernel_name}${kernel_cpu_suffix}${blob_data_suffix}")
    set(target_blob_text_path "${CMAKE_BINARY_DIR}/${target_blob_text_file}")
    set(target_blob_data_path "${CMAKE_BINARY_DIR}/${target_blob_data_file}")
    message(STATUS "blob_text = ${target_blob_text_path}")
    message(STATUS "blob_data = ${target_blob_data_path}")
  endif()

  if(BUILD_JTAG_BINARIES)
    set(target_jtag_text_file "${kernel_name}${kernel_cpu_suffix}${jtag_text_suffix}")
    set(target_jtag_data_file "${kernel_name}${kernel_cpu_suffix}${jtag_data_suffix}")
    set(target_jtag_text_path "${CMAKE_BINARY_DIR}/${target_jtag_text_file}")
    set(target_jtag_data_path "${CMAKE_BINARY_DIR}/${target_jtag_data_file}")
    message(STATUS "jtag_text = ${target_jtag_text_path}")
    message(STATUS "jtag_data = ${target_jtag_data_path}")
  endif()
endforeach()

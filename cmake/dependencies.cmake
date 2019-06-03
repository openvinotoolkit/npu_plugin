# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(models)

#we have number of dependencies stored on ftp
include(dependency_solver)

set_temp_directory(TEMP "${IE_MAIN_SOURCE_DIR}")

include(ExternalProject)

if (ENABLE_SAME_BRANCH_FOR_MODELS)
    branchName(MODELS_BRANCH)
else()
    set(MODELS_BRANCH "master")
endif()

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=" ${MODELS_PATH})

#validation set repository added always by default
#add_validation_set_repo(${ENABLE_FUNCTIONAL_TESTS}  "validation_set:inference-engine/validation-set.git")

add_models_repo(${ENABLE_MODELS}                "models:inference-engine/models-ir.git")
add_models_repo(${ENABLE_PRIVATE_MODELS}        "models_private:inference-engine-models/private-ir.git")
add_models_repo(${ENABLE_PRIVATE_HDDL_MODELS}   "models_hddl_private:hddl/hddl-private-ir.git")
add_models_repo(${ENABLE_MODELS_FOR_CVSDK}      "models-for-cvsdk:inference-engine-models/models-for-cvsdk.git" "ie_regression")

fetch_models_and_validation_set()

include(linux_name)
if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if (ENABLE_MYRIAD OR ENABLE_KMB)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2450
            ARCHIVE_UNIFIED VPU/ma2450/firmware_ma2450_571.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2450"
            ENVIRONMENT "VPU_FIRMWARE_MA2450"
            FOLDER)
    debug_message(STATUS "ma2450=" ${VPU_FIRMWARE_MA2450})
endif ()

# TODO: for now, use universal FW only for myriad plugin
# (keep it also for kmb plugin just to be able to build it)
if (ENABLE_MYRIAD OR ENABLE_KMB)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2X8X
            ARCHIVE_UNIFIED VPU/ma2x8x/firmware_ma2x8x_571.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2x8x"
            ENVIRONMENT "VPU_FIRMWARE_MA2x8x"
            FOLDER)
    debug_message(STATUS "ma2x8x=" ${VPU_FIRMWARE_MA2x8x})
endif ()

if (ENABLE_HDDL)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2480
            ARCHIVE_UNIFIED VPU/ma2480/firmware_ma2480_571.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2480"
            ENVIRONMENT "VPU_FIRMWARE_MA2480"
            FOLDER)
    debug_message(STATUS "ma2480=" ${VPU_FIRMWARE_MA2480})

    if (WIN32)
        RESOLVE_DEPENDENCY(HDDL
                ARCHIVE_WIN hddl/hddl_windows10_552.zip
                TARGET_PATH "${TEMP}/vpu/hddl"
                ENVIRONMENT "HDDL")
    elseif(LINUX)
        if (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
            RESOLVE_DEPENDENCY(HDDL
                    ARCHIVE_LIN hddl/hddl_ubuntu18_566.tar.gz
                    TARGET_PATH "${TEMP}/vpu/hddl"
                    ENVIRONMENT "HDDL")
        elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
            RESOLVE_DEPENDENCY(HDDL
                    ARCHIVE_LIN hddl/hddl_ubuntu16_566.tar.gz
                    TARGET_PATH "${TEMP}/vpu/hddl"
                    ENVIRONMENT "HDDL")
        elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7")
            RESOLVE_DEPENDENCY(HDDL
                    ARCHIVE_LIN hddl/hddl_centos7_566.tar.gz
                    TARGET_PATH "${TEMP}/vpu/hddl"
                    ENVIRONMENT "HDDL")
        endif()
    endif()
    debug_message(STATUS "hddl=" ${HDDL})
endif ()

if (ENABLE_DLIA)
    if (WIN32)
        RESOLVE_DEPENDENCY(AOCL_RTE
                ARCHIVE_WIN "aoclrte-windows64_18.1.1.263.zip"
                TARGET_PATH "${TEMP}/aocl-rte"
                VERSION_REGEX ".*_(([a-z]+-)?[a-z]+-[0-9]+)---.*"
                ENVIRONMENT "AOCL_RTE")
        log_rpath_from_dir(AOCL_RTE "aocl-rte\\windows64\\bin")
    elseif(LINUX)
        RESOLVE_DEPENDENCY(AOCL_RTE
                ARCHIVE_LIN "aocl-rte_18.1.1.263p.tar.gz"
                TARGET_PATH "${TEMP}/aocl-rte"
                VERSION_REGEX ".*_(([a-z]+-)?[a-z]+-[0-9]+)---.*"
                ENVIRONMENT "AOCL_RTE")
        log_rpath_from_dir(AOCL_RTE "aocl-rte/linux64/lib")
    endif()
    message(STATUS "AOCL_RTE=" ${AOCL_RTE})
endif ()

## enable cblas_gemm from OpenBLAS package
if (GEMM STREQUAL "OPENBLAS")
if(NOT BLAS_LIBRARIES OR NOT BLAS_INCLUDE_DIRS)
    find_package(BLAS REQUIRED)
    if(BLAS_FOUND)
        find_path(BLAS_INCLUDE_DIRS cblas.h)
    else()
        message(ERROR "OpenBLAS not found: install OpenBLAS or set -DBLAS_INCLUDE_DIRS=<path to dir with cblas.h> and -DBLAS_LIBRARIES=<path to libopenblas.so or openblas.lib>")
    endif()
endif()
debug_message(STATUS "openblas=" ${BLAS_LIBRARIES})
endif ()

## enable cblas_gemm from MKL-ml package
if (GEMM STREQUAL "MKL")
if (WIN32)
    #TODO: add target_path to be platform specific as well, to avoid following if
    RESOLVE_DEPENDENCY(MKL
            ARCHIVE_WIN "mkltiny_win_20190415.zip"
            TARGET_PATH "${TEMP}/mkltiny_win_20190415"
            ENVIRONMENT "MKLROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
elseif(LINUX)
    RESOLVE_DEPENDENCY(MKL
            ARCHIVE_LIN "mkltiny_lnx_20190131.tgz"
            TARGET_PATH "${TEMP}/mkltiny_lnx_20190131"
            ENVIRONMENT "MKLROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
else(APPLE)
    RESOLVE_DEPENDENCY(MKL
            ARCHIVE_MAC "mkltiny_mac_20190414.tgz"
            TARGET_PATH "${TEMP}/mkltiny_mac_20190414"
            ENVIRONMENT "MKLROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
endif()
debug_message(STATUS "mkl_ml=" ${MKL})
endif ()

## Intel OMP package
if (THREADING STREQUAL "OMP")
if (WIN32)
    RESOLVE_DEPENDENCY(OMP
            ARCHIVE_WIN "iomp.zip"
            TARGET_PATH "${TEMP}/omp"
            ENVIRONMENT "OMP"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
elseif(LINUX)
    RESOLVE_DEPENDENCY(OMP
            ARCHIVE_LIN "iomp.tgz"
            TARGET_PATH "${TEMP}/omp"
            ENVIRONMENT "OMP"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
else(APPLE)
    RESOLVE_DEPENDENCY(OMP
            ARCHIVE_MAC "iomp_20190130_mac.tgz"
            TARGET_PATH "${TEMP}/omp"
            ENVIRONMENT "OMP"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
endif()
log_rpath_from_dir(OMP "${OMP}/lib")
debug_message(STATUS "intel_omp=" ${OMP})
endif ()

## TBB package
if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
if (WIN32)
    #TODO: add target_path to be platform specific as well, to avoid following if
    RESOLVE_DEPENDENCY(TBB
            ARCHIVE_WIN "tbb2019_20181010_win.zip" #TODO: windows zip archive created incorrectly using old name for folder
            TARGET_PATH "${TEMP}/tbb"
            ENVIRONMENT "TBBROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
elseif(LINUX)
    RESOLVE_DEPENDENCY(TBB
            ARCHIVE_LIN "tbb2019_20181010_lin.tgz"
            TARGET_PATH "${TEMP}/tbb"
            ENVIRONMENT "TBBROOT")
else(APPLE)
    RESOLVE_DEPENDENCY(TBB
            ARCHIVE_MAC "tbb2019_20190414_mac.tgz"
            TARGET_PATH "${TEMP}/tbb"
            ENVIRONMENT "TBBROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
endif()
log_rpath_from_dir(TBB "${TBB}/lib")
debug_message(STATUS "tbb=" ${TBB})
endif ()

if (ENABLE_OPENCV)
  set(OPENCV_VERSION "4.1.0")
  set(OPENCV_BUILD "0506")
  set(OPENCV_SUFFIX "")
if (WIN32)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_WIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.zip"
            TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "\\opencv_${OPENCV_VERSION}\\bin")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
elseif(APPLE)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_MAC "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.tar.xz"
            TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_${OPENCV_VERSION}_osx/lib")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
elseif(LINUX)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(OPENCV_SUFFIX "yocto_kmb")
    elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
        set(OPENCV_SUFFIX "ubuntu16")
    elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
        set(OPENCV_SUFFIX "ubuntu18")
    elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7")
        set(OPENCV_SUFFIX "centos7")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l" AND
            (${LINUX_OS_NAME} STREQUAL "Debian 9" OR
             ${LINUX_OS_NAME} STREQUAL "Raspbian 9"))
        set(OPENCV_SUFFIX "debian9arm")
    endif()
endif()

if (OPENCV_SUFFIX)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.tar.xz"
            TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/lib")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
endif()

debug_message(STATUS "opencv=" ${OPENCV})
set(OpenCV_DIR "${OPENCV}" CACHE PATH "Path to OpenCV in temp directory")
endif()

include(ie_parallel)

if (ENABLE_INTEGRATION_TESTS)
    #Model Optimizer
    RESOLVE_DEPENDENCY(MODELOPTIMIZER_BIN_DIR
            ARCHIVE utils/mo_linux_x64_release.tar.bz2
            TARGET_PATH "${TEMP}/utils/ModelOptimizer"
            ENVIRONMENT "MODELOPTIMIZER_BIN_DIR")
endif()

if (ENABLE_GNA)
    RESOLVE_DEPENDENCY(GNA
            ARCHIVE_UNIFIED "GNA/gna_20181120.zip"
            TARGET_PATH "${TEMP}/gna")
endif()

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/share/InferenceEngineConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
        @ONLY)

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        COPYONLY)

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/ie_parallel.cmake"
        "${CMAKE_BINARY_DIR}/share/ie_parallel.cmake"
        COPYONLY)

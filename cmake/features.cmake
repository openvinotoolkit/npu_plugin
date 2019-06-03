# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include (options)

#this options are aimed to optimize build time on development system

#backed targets

ie_option (ENABLE_GNA "GNA support for inference engine" ON)

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON)

ie_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON)

ie_option (ENABLE_CLDNN_TESTS "Enable clDNN unit tests" OFF)

ie_option (ENABLE_CLDNN_BUILD "build clDnn from sources" OFF)

ie_option (ENABLE_PROFILING_ITT "ITT tracing of IE and plugins internals" ON)

ie_option (ENABLE_PROFILING_RAW "Raw counters profiling (just values, no start/stop time or timeline)" OFF)

# "MKL-DNN library might use MKL-ML or OpenBLAS for gemm tasks: MKL|OPENBLAS|JIT"
if (NOT GEMM STREQUAL "MKL"
        AND NOT GEMM STREQUAL "OPENBLAS"
        AND NOT GEMM STREQUAL "JIT")
    set (GEMM "MKL")
    message(STATUS "GEMM should be set to MKL, OPENBLAS or JIT. Default option is " ${GEMM})
endif()
set(GEMM "${GEMM}" CACHE STRING "Gemm implementation" FORCE)
list (APPEND IE_OPTIONS GEMM)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if (NOT THREADING STREQUAL "TBB"
        AND NOT THREADING STREQUAL "TBB_AUTO"
        AND NOT THREADING STREQUAL "OMP"
        AND NOT THREADING STREQUAL "SEQ")
    set (THREADING "TBB")
    message(STATUS "THREADING should be set to TBB, TBB_AUTO, OMP or SEQ. Default option is " ${THREADING})
endif()
set(THREADING "${THREADING}" CACHE STRING "Threading" FORCE)
list (APPEND IE_OPTIONS THREADING)

ie_option (ENABLE_DOCKER "docker images" OFF)

ie_option (ENABLE_DLIA "dlia based plugin for inference engine" ON)

ie_option (ENABLE_VPU "vpu targeted plugins for inference engine" ON)

ie_option (ENABLE_MYRIAD "myriad targeted plugin for inference engine" ON)

ie_option (ENABLE_MYRIAD_NO_BOOT "myriad plugin will skip device boot" OFF)

ie_option (ENABLE_HDDL "hddl targeted plugin for inference engine" ON)

ie_option (ENABLE_KMB "kmb targeted plugin for inference engine" OFF)

ie_option (ENABLE_TESTS "unit and functional tests" OFF)

ie_option (ENABLE_GAPI_TESTS "tests for GAPI kernels" OFF)

ie_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF)

ie_option (ENABLE_MYRIAD_MVNC_TESTS "functional and behavior tests for mvnc api" OFF)

ie_option (ENABLE_MODELS "download all models required for functional testing" ON)

ie_option (ENABLE_VALIDATION_SET "download validation_set required for functional testing" ON)

ie_option (ENABLE_PRIVATE_MODELS "fetch models from private repo" OFF)

ie_option (ENABLE_PRIVATE_HDDL_MODELS "fetch models from hddl-private-ir repo" OFF)

ie_option (ENABLE_MODELS_FOR_CVSDK "fetch models from models-for-cvsdk repo" OFF)

ie_option (ENABLE_SAME_BRANCH_FOR_MODELS "uses same branch for models and for inference engine, if not enabled models are taken from master" OFF)

ie_option (ENABLE_BEH_TESTS "tests oriented to check inference engine API corecteness" ON)

ie_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON)

ie_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON)

ie_option (ENABLE_SAMPLES_CORE "console samples core library" ON)

ie_option (ENABLE_SERVICE_AGENT "service web agent" ON)

ie_option (ENABLE_SANITIZER "enable checking memory errors via AddressSanitizer" OFF)

ie_option (ENABLE_FUZZING "instrument build for fuzzing" OFF)

ie_option (COVERAGE "enable code coverage" OFF)

ie_option (ENABLE_INTEGRATION_TESTS "integration tests" OFF)

ie_option (ENABLE_STRESS_UNIT_TESTS "stress unit tests" OFF)

ie_option (VERBOSE_BUILD "shows extra information about build" OFF)

ie_option (ENABLE_UNSAFE_LOCATIONS "skip check for MD5 for dependency" OFF)

ie_option (ENABLE_ALTERNATIVE_TEMP "in case of dependency conflict, to avoid modification in master, use local copy of dependency" ON)

ie_option (ENABLE_SEGMENTATION_TESTS "segmentation tests" ON)

ie_option (ENABLE_OBJECT_DETECTION_TESTS "object detection tests" ON)

ie_option (ENABLE_ICV_TESTS "function tests that include ICV topologies" ON)

ie_option (ENABLE_DUMP "enables mode for dumping per layer information" OFF)

ie_option (ENABLE_OPENCV "enables OpenCV" ON)

ie_option (OS_FOLDER "create OS dedicated folder in output" OFF)

ie_option (ENABLE_PLUGIN_RPATH "enables rpath information to be present in plugins binary, and in corresponding test_applications" ON)

ie_option (ENABLE_AFFINITY_GENERATOR "enables affinity generator build" OFF)

ie_option (ENABLE_DEBUG_SYMBOLS "generates symbols for debugging" OFF)

ie_option (ENABLE_PYTHON "enables ie python bridge build" OFF)

ie_option (DEVELOPMENT_PLUGIN_MODE "Disabled build of all plugins" OFF)

ie_option (PRUNE_LFS_MODELS "Prune LFS cache for models and validation-set" OFF)

ie_option (TREAT_WARNING_AS_ERROR "Treat build warnings as errors" ON)

if (UNIX AND NOT APPLE)
    ie_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ON)
    ie_option(ENABLE_CPPLINT_REPORT "Build cpplint report instead of failing the build" OFF)
else()
    set(ENABLE_CPPLINT OFF)
endif()

if (UNIX AND NOT APPLE AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.10)
    ie_option(ENABLE_CPPCHECK "Enable cppcheck during the build" ON)
else()
    set(ENABLE_CPPCHECK OFF)
endif()

#environment variables used

#name of environment variable stored path to temp directory"
set (DL_SDK_TEMP  "DL_SDK_TEMP")

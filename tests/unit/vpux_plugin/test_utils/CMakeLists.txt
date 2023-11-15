#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

# throw_test_backend
# SHARED or STATIC depending on BUILD_SHARED_LIBS
add_library("throw_test_backend" "${CMAKE_CURRENT_SOURCE_DIR}/throw_test_backend.cpp")
target_link_libraries("throw_test_backend" PRIVATE npu_al IE::inference_engine_plugin_api npu_utils)
target_compile_definitions("throw_test_backend" PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)
enable_warnings_as_errors("throw_test_backend")

# vpu3700_test_backend
# SHARED or STATIC depending on BUILD_SHARED_LIBS
add_library("vpu3700_test_backend" "${CMAKE_CURRENT_SOURCE_DIR}/vpu3700_test_backend.cpp")
target_link_libraries("vpu3700_test_backend" PRIVATE npu_al IE::inference_engine_plugin_api npu_utils)
target_compile_definitions("vpu3700_test_backend" PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)
enable_warnings_as_errors("vpu3700_test_backend")

# no_devices_test_backend
# SHARED or STATIC depending on BUILD_SHARED_LIBS
add_library("no_devices_test_backend" "${CMAKE_CURRENT_SOURCE_DIR}/no_devices_test_backend.cpp")
target_link_libraries("no_devices_test_backend" PRIVATE npu_al IE::inference_engine_plugin_api npu_utils)
target_compile_definitions("no_devices_test_backend" PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)
enable_warnings_as_errors("no_devices_test_backend")

# vpu3720_test_backend
# SHARED or STATIC depending on BUILD_SHARED_LIBS
add_library("vpu3720_test_backend" "${CMAKE_CURRENT_SOURCE_DIR}/vpu3720_test_backend.cpp")
target_link_libraries("vpu3720_test_backend" PRIVATE npu_al IE::inference_engine_plugin_api npu_utils)
target_compile_definitions("vpu3720_test_backend" PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)
enable_warnings_as_errors("vpu3720_test_backend")

vpux_enable_clang_format("throw_test_backend" "vpu3700_test_backend" "no_devices_test_backend" "vpu3720_test_backend")

#
# targets install
#

set(TEST_BACKENDS "throw_test_backend" "vpu3700_test_backend" "no_devices_test_backend" "vpu3720_test_backend")

if (BUILD_SHARED_LIBS)
    install(TARGETS ${TEST_BACKENDS} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT tests EXCLUDE_FROM_ALL

            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH}
            COMPONENT tests EXCLUDE_FROM_ALL

            LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT tests EXCLUDE_FROM_ALL
    )
else()
    install(TARGETS ${TEST_BACKENDS}
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT tests EXCLUDE_FROM_ALL

            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH}
            COMPONENT tests EXCLUDE_FROM_ALL

            LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT tests EXCLUDE_FROM_ALL
    )
endif()

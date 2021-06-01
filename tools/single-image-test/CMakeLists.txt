#
# Copyright 2020 Intel Corporation.
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

set(TARGET_NAME single-image-test)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
find_package(gflags QUIET)
find_package(Python3 COMPONENTS Interpreter QUIET)

if(NOT OpenCV_FOUND OR NOT gflags_FOUND)
    message(WARNING "${TARGET_NAME} tool is disabled due to missing dependencies")
    return()
endif()

add_executable(${TARGET_NAME} main.cpp utils.cpp run_movisim_emulator.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    IE::inference_engine IE::ngraph
    IE::inference_engine_plugin_api
    opencv_core opencv_imgproc opencv_imgcodecs
    gflags
    kmb_utils
)

if(NOT Python3_FOUND)
    message(WARNING "Python3 is not found, run single-image-tests via MoviSim will not aviable")
else()
    message(WARNING "Python3 is found, run single-image-tests via MoviSim is aviable")
    target_compile_definitions(${TARGET_NAME}
        PUBLIC
            ENABLE_MOVISIM
        PUBLIC
            Python_EXECUTABLE="${Python3_EXECUTABLE}"
    )
    message(WARNING "Copy run_MoviSim python script to: " ${OpenVINO_MAIN_SOURCE_DIR}${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    add_custom_command(
        TARGET  ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/scripts/run_MoviSim/run_MoviSim.py
                ${OpenVINO_MAIN_SOURCE_DIR}${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/run_MoviSim.py)
endif()


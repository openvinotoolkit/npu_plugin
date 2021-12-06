// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet"\
                                    "and a .bmp file for the other networks.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). " \
                                            "Default value is CPU. Sample will look for a suitable plugin for device specified.";

/// @brief message for top results number
static const char ntop_message[] = "Optional. Number of top results. Default value is 10.";

/// @brief message for plugin messages
static const char plugin_message[] = "Optional. Enables messages from a plugin";

/// @brief message for color order
static const char rgb_message[] = "Optional. Use input image in RGB format. Default is BGR.";

/// @brief message for input precision
static const char input_precision_message[] = "Optional. u8, f32 or f16. Default u8. ";

/// @brief message for input precision
static const char output_precision_message[] = "Optional. u8, f32 or f16. Default fp16. ";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Top results number (default 10) <br>
DEFINE_uint32(nt, 10, ntop_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_message);

/// @brief Define parameter for color <br>
/// It is an optional parameter
DEFINE_bool(r, false, rgb_message);

/// @brief Define input precision <br>
/// It is an optional parameter
DEFINE_string(ip, "u8", input_precision_message);

/// @brief Define output precision <br>
/// It is an optional parameter
DEFINE_string(op, "f32", output_precision_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "test_classification [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -nt \"<integer>\"         " << ntop_message << std::endl;
    std::cout << "    -p_msg                  " << plugin_message << std::endl;
    std::cout << "    -ip_msg                  " << input_precision_message << std::endl;
    std::cout << "    -op_msg                  " << output_precision_message << std::endl;
}


//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

const int DEFAULT_ZERO_POINT = 0;
const float DEFAULT_SCALE = 1.0;

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a binary input file";

/// @brief message for model argument
static const char model_message[] = "Required. Path to a .blob file compiled from .xml file with a trained model.";

/// @brief message for plugin messages
static const char plugin_message[] = "Enables messages from a plugin";

/// @brief message for scale
static const char scale_message[] = "Scale of output from fully connected layer. Default value is 1.0";

/// @brief message for zero point
static const char zeropoint_message[] = "Zero point of output from fully connected layer. Default value is 0";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief message for scale
DEFINE_double(s, DEFAULT_SCALE, scale_message);

/// @brief message for zero point
DEFINE_int32(z, DEFAULT_ZERO_POINT, zeropoint_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "kmb_classification_sample [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"           " << image_message << std::endl;
    std::cout << "    -m \"<path>\"           " << model_message << std::endl;
    std::cout << "    -s value                " << scale_message << std::endl;
    std::cout << "    -z zeropoint            " << zeropoint_message << std::endl;
}

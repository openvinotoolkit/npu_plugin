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

/// @brief message for blob argument
static const char blob_message[] = "Required. Path to the blob to query zero point and scale";

/// @brief message for tensor argument
static const char a_tensor_message[] = "Required. Path to the actual results tensor to be converted.";

/// @brief message for tensor argument
static const char e_tensor_message[] = "Required. Path to the expected results tensor.";

/// @brief message for data type
static const char dtype_message[] = "Optional. Datatype to convert to, eg, U8 | FP32 only supported. Default is FP32";

/// @brief message for setting quantize or not
static const char quantize_message[] = "Optional. Default is to de-quantize. Specify --quantize if you wish to quantize";

/// @brief message for setting quantize or not
static const char tolerence_message[] = "Optional. Tolerence use in comparing floating point numbers, default is 0.0f";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set blob file <br>
/// It is a required parameter
DEFINE_string(b, "", blob_message);

/// @brief Define parameter for set tensor file <br>
/// It is a required parameter
DEFINE_string(a, "", a_tensor_message);

/// @brief Define parameter for set tensor file <br>
/// It is a required parameter
DEFINE_string(e, "", e_tensor_message);

/// @brief Define the datatpye <br>
DEFINE_string(d, "FP16", dtype_message);

/// @brief Define quantize message
DEFINE_bool(q, false, quantize_message);

/// @brief Define tolerence message
DEFINE_double(t, 0.0f, tolerence_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "convert_tensor [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                    " << help_message << std::endl;
    std::cout << "    -b \"<path>\"           " << blob_message << std::endl;
    std::cout << "    -a \"<path>\"           " << a_tensor_message << std::endl;
    std::cout << "    -e \"<path>\"           " << e_tensor_message << std::endl;
    std::cout << "    -q true|false         " << quantize_message << std::endl;
    std::cout << "    -d U8|FP16            " << dtype_message << std::endl;
    std::cout << "    -t <float>            " << tolerence_message << std::endl;
}

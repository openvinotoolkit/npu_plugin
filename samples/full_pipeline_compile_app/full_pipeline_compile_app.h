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

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). " \
                                            "Default value is CPU. Sample will look for a suitable plugin for device specified.";

/// @brief message for top results number
static const char ntop_message[] = "Optional. Number of top results. Default value is 10.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers." \
                                                 "Absolute path to a shared library with the kernels implementation";

/// @brief message for plugin messages
static const char plugin_message[] = "Optional. Enables messages from a plugin";

/// @brief message for model blob argument
static const char blob_message[] = "Optional. Path to a .blob of main part ('body' subnetwork) file compiled from .xml file.";

/// @brief message for model blob argument
static const char split_list_message[] = "Required. Path to a file with list of layers to split network.";

/// @brief message for 'out' path argument
static const char out_message[] = "Optional. Path to place subnetworks (head, body and tail).";

/// @brief message for no_quantize argument
static const char no_quantize_message[] = "Optional. Bool. If set original network is not quantized before splitting.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief Top results number (default 10) <br>
DEFINE_uint32(nt, 10, ntop_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_message);

/// @brief Define parameter for set out path <br>
/// It is an optional parameter
DEFINE_string(out, "", out_message);

/// @brief Define parameter for set blob path <br>
/// It is an optional parameter
DEFINE_string(blob, "", blob_message);

/// @brief Define parameter for set split network list file <br>
/// It is a required parameter
DEFINE_string(split, "", split_list_message);

/// @brief Define parameter for switch of quantization
/// It is a required parameter
DEFINE_bool(no_quantize, false, no_quantize_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "full_pipeline_compile_app [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -d \"<device>\"         " << target_device_message << std::endl;
    std::cout << "    -m \"<path>\"           " << model_message << std::endl;
    std::cout << "    -out \"<path>\"         " << out_message << std::endl;
    std::cout << "    -blob \"<path>\"        " << blob_message << std::endl;
    std::cout << "    -split \"<path>\"       " << split_list_message << std::endl;
    std::cout << "    -no_quantize            " << no_quantize_message << std::endl;
}


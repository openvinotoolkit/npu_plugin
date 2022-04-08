//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <vector>

/// @brief message for help argument
static const char help_message[] = "Show this help message.";

/// @brief message for width argument
static const char width_message[] = "(Optional) Specify the input width resolution.(Default 1280)";

/// @brief message for height argument
static const char height_message[] = "(Optional) Specify the input height resolution.(Default 720)";

/// @brief message for output messages
static const char output_message[] =
        "(Optional) Path to the folder where to generate de IR (.xml and .bin).(default current folder)";

/// @brief message for json argument
static const char json_message[] = "(Required) Path to json file that contains weights and biases.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
DECLARE_bool(help);

/// @brief Define parameter for set width <br>
/// It is a required parameter
DEFINE_int32(width, 1280, width_message);

/// @brief Define parameter for set height <br>
/// It is a required parameter
DEFINE_int32(height, 720, height_message);

/// @brief Define parameter for set output folder <br>
DEFINE_string(o, ".", output_message);

/// @brief Define parameter for set json <br>
/// It is a required parameter
DEFINE_string(json, "", json_message);

/**
 * @brief This function show a help message
 */

static void showUsage(const std::string& name) {
    std::cerr << "Usage: " << name << " <option(s)> \n"
              << "Options:\n"
              << "    -h,--help                    " << help_message << "\n"
              << "    --width WIDTH                " << width_message << "\n"
              << "    --height HEIGHT              " << height_message << "\n"
              << "    --o PATH_TO_OUTPUT_FOLDER    " << output_message << "\n"
              << "    --json PATH_TO_JSON_FILE     " << json_message << "\n"
              << std::endl;
}

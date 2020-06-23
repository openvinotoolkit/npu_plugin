//
// Copyright (C) 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <gflags/gflags.h>

#include "inference_engine.hpp"
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/vpu_compiler_config.hpp>
#include "samples/common.hpp"
#include <vpu/utils/string.hpp>

#include "vpu_tools_common.hpp"

static constexpr char help_message[] = "Optional. Prints a usage message.";
static constexpr char model_message[] = "Required. File containing xml model.";
static constexpr char plugin_path_message[] = "Optional. Plugin folder.";
static constexpr char output_message[] = "Optional. Output blob file. Default value: \"<model_xml_file>.blob\".";
static constexpr char config_message[] = "Optional. Key-value configuration text file. Default value: \"config\".";
static constexpr char platform_message[] = "Optional. Specifies movidius platform."
                                           " Supported values: VPU_2490."
                                           " Overwrites value from config.\n"
"                                             This option might be used in order to compile blob"
                                           " without VPU device connected.";
static constexpr char inputs_precision_message[] = "Optional. Specifies precision for all input layers of network."
                                                   " Supported values: FP16, U8. Default value: U8.";
static constexpr char outputs_precision_message[] = "Optional. Specifies precision for all output layers of network."
                                                    " Supported values: FP16, U8. Default value: FP32.";
static constexpr char iop_message[] = "Optional. Specifies precision for input/output layers by name.\n"
"                                             By default all input layers have U8 precision,\n"
"                                             all output layers have FP32 precision.\n"
"                                             Available precisions: FP16, U8.\n"
"                                             Example: -iop \"input:FP16, output:FP16\".\n"
"                                             Notice that quotes are required.\n"
"                                             Overwrites precision from ip and op options for specified layers.";
static constexpr char mcm_target_descriptor_message[] = "Optional. Compilation target descriptor file.";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", model_message);
DEFINE_string(pp, "", plugin_path_message);
DEFINE_string(o, "", output_message);
DEFINE_string(c, "config", config_message);
DEFINE_string(ip, "", inputs_precision_message);
DEFINE_string(op, "", outputs_precision_message);
DEFINE_string(iop, "", iop_message);
DEFINE_string(VPU_PLATFORM, "", platform_message);
DEFINE_string(TARGET_DESCRIPTOR, "", mcm_target_descriptor_message);

static const InferenceEngine::Precision defaultInputPrecision = InferenceEngine::Precision::U8;
static const InferenceEngine::Precision defaultOutputPrecision = InferenceEngine::Precision::FP32;

static void showUsage() {
    std::cout << std::endl;
    std::cout << "vpu2_compile [OPTIONS]" << std::endl;
    std::cout << "[OPTIONS]:" << std::endl;
    std::cout << "    -h                                       "   << help_message                 << std::endl;
    std::cout << "    -m                           <value>     "   << model_message                << std::endl;
    std::cout << "    -pp                          <value>     "   << plugin_path_message          << std::endl;
    std::cout << "    -o                           <value>     "   << output_message               << std::endl;
    std::cout << "    -c                           <value>     "   << config_message               << std::endl;
    std::cout << "    -ip                          <value>     "   << inputs_precision_message     << std::endl;
    std::cout << "    -op                          <value>     "   << outputs_precision_message    << std::endl;
    std::cout << "    -iop                        \"<value>\"    " << iop_message                  << std::endl;
    std::cout << "    -TARGET_DESCRIPTOR           <value>     "   << mcm_target_descriptor_message << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int *argc, char ***argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    gflags::CommandLineFlagInfo m;
    if (!gflags::GetCommandLineFlagInfo("m", &m) || m.is_default) {
            throw std::invalid_argument("-m is required option");
    }

    if (1 < *argc) {
        char ** const args = *argv;
        std::ostringstream message;
        message << "Unknown arguments supplied: ";
        for (auto arg = 1; arg < *argc; arg++) {
            message << args[arg];
            if (arg < *argc) {
                message << " ";
            }
        }
        throw std::invalid_argument(message.str());
    }

    return true;
}

static std::map<std::string, std::string> configure(const std::string &configFile, const std::string &xmlFileName) {
    auto config = parseConfig(configFile);

    if (!FLAGS_VPU_PLATFORM.empty()) {
        config[VPU_KMB_CONFIG_KEY(PLATFORM)] = FLAGS_VPU_PLATFORM;
    }

    if (!FLAGS_TARGET_DESCRIPTOR.empty()) {
        config[VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR)] = FLAGS_TARGET_DESCRIPTOR;
    }

    return config;
}

static std::map<std::string, std::string> parsePrecisions(const std::string &iop) {
    std::string user_input = iop;
    user_input.erase(std::remove_if(user_input.begin(), user_input.end(), ::isspace), user_input.end());

    std::vector<std::string> inputs;
    vpu::splitStringList(user_input, inputs, ',');

    std::map<std::string, std::string> precisions;
    for (auto &&input : inputs) {
        std::vector<std::string> precision;
        vpu::splitStringList(input, precision, ':');
        if (precision.size() != 2) {
            throw std::invalid_argument("Invalid precision " + input + ". Expected layer_name : precision_value");
        }

        precisions[precision[0]] = precision[1];
    }

    return precisions;
}

using supported_precisions_t = std::unordered_map<std::string, InferenceEngine::Precision>;

static InferenceEngine::Precision getPrecision(const std::string &value,
                                               const supported_precisions_t &supported_precisions,
                                               const std::string& error_report = std::string()) {
    std::string upper_value = value;
    std::transform(value.begin(), value.end(), upper_value.begin(), ::toupper);
    auto precision = supported_precisions.find(upper_value);
    if (precision == supported_precisions.end()) {
        std::string report = error_report.empty() ? ("") : (" " + error_report);
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision" + report);
    }

    return precision->second;
}

static InferenceEngine::Precision getInputPrecision(const std::string &value) {
    static const supported_precisions_t supported_precisions = {
         { "FP16", InferenceEngine::Precision::FP16 },
         { "U8",   InferenceEngine::Precision::U8 }
    };
    static const InferenceEngine::Precision ip =
            getPrecision(value, supported_precisions, "for input layer");
    return ip;
}

static InferenceEngine::Precision getOutputPrecision(const std::string &value) {
    static const supported_precisions_t supported_precisions = {
            { "FP32", InferenceEngine::Precision::FP32 },
            { "FP16", InferenceEngine::Precision::FP16 },
            { "U8",   InferenceEngine::Precision::U8 }
    };
    static const InferenceEngine::Precision op =
        getPrecision(value, supported_precisions, "for output layer");
    return op;
}

void setPrecisions(const InferenceEngine::CNNNetwork &network, const std::string &iop) {
    auto precisions = parsePrecisions(iop);
    auto inputs = network.getInputsInfo();
    auto outputs = network.getOutputsInfo();

    for (auto &&layer : precisions) {
        auto name = layer.first;

        auto input_precision = inputs.find(name);
        auto output_precision = outputs.find(name);

        if (input_precision != inputs.end()) {
            input_precision->second->setPrecision(getInputPrecision(layer.second));
        } else if (output_precision != outputs.end()) {
            output_precision->second->setPrecision(getOutputPrecision(layer.second));
        } else {
            throw std::logic_error(name + " is not an input neither output");
        }
    }
}

static void processPrecisions(InferenceEngine::CNNNetwork &network,
                              const std::string &inputs_precision, const std::string &outputs_precision,
                              const std::string &iop) {
    const auto in_precision = inputs_precision.empty() ? defaultInputPrecision
                                                 : getInputPrecision(inputs_precision);
    for (auto &&layer : network.getInputsInfo()) {
        layer.second->setLayout(InferenceEngine::Layout::NHWC);
        layer.second->setPrecision(in_precision);
    }

    const auto out_precision = outputs_precision.empty() ? defaultOutputPrecision
            : getOutputPrecision(outputs_precision);
    for (auto &&layer : network.getOutputsInfo()) {
        if (layer.second->getDims().size() == 2) {
            layer.second->setLayout(InferenceEngine::Layout::NC);
        } else {
            layer.second->setLayout(InferenceEngine::Layout::NHWC);
        }
        layer.second->setPrecision(out_precision);
    }

    if (!iop.empty()) {
        setPrecisions(network, iop);
    }
}

using Time = std::chrono::time_point<std::chrono::steady_clock>;
using TimeDiff = std::chrono::milliseconds;
Time (*GetCurrentTime)() = &std::chrono::steady_clock::now;

int main(int argc, char *argv[]) {
    TimeDiff loadNetworkTimeElapsed {0};
    try {
        std::cout << "Inference Engine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }

        auto network = readNetwork(FLAGS_m);

        processPrecisions(network, FLAGS_ip, FLAGS_op, FLAGS_iop);

        InferenceEngine::Core ie;
        Time timeBeforeLoadNetwork = GetCurrentTime();
        auto executableNetwork = ie.LoadNetwork(network, "KMB", configure(FLAGS_c, FLAGS_m));
        loadNetworkTimeElapsed = std::chrono::duration_cast<TimeDiff>(GetCurrentTime() - timeBeforeLoadNetwork);

        std::string outputName = FLAGS_o;
        if (outputName.empty()) {
            outputName = fileNameNoExt(FLAGS_m) + ".blob";
        }
        executableNetwork.Export(outputName);
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << std::endl << "Try running with -h for help message" << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Done. LoadNetwork time elapsed: " << loadNetworkTimeElapsed.count() << " ms" << std::endl;
    return EXIT_SUCCESS;
}

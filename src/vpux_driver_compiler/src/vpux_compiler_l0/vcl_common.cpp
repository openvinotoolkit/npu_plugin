//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_common.hpp"

#include <istream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>

#include "vcl_compiler.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/vpux_plugin_config.hpp"

/**
 * @name Key of build flags
 * @{
 */
#define KEY_INPUTS_PRECISIONS "--inputs_precisions"
#define KEY_INPUTS_LAYOUTS "--inputs_layouts"
#define KEY_INPUTS_MODEL_LAYOUTS "--inputs_model_layouts"
#define KEY_OUTPUTS_PRECISIONS "--outputs_precisions"
#define KEY_OUTPUTS_LAYOUTS "--outputs_layouts"
#define KEY_OUTPUTS_MODEL_LAYOUTS "--outputs_model_layouts"
/// The seperator of input output info and compilation configs
#define KEY_CONFIGS "--config"
/** @} */

using namespace vpux;

namespace VPUXDriverCompiler {

/**
 * @brief Parse single option and create map to save the key and value
 *
 * @tparam T The type of value
 * @param option The content may like $KEY_INPUTS_PRECISIONS="InputName:InputPrecision ...", content depend on the type
 * of key
 * @param vclLogger The logger of current compiler
 * @param results The arrays to store parsed results
 * @param function The function to convert string value to the T type
 * @return vcl_result_t
 */
template <typename T>
vcl_result_t parseSingleOption(const std::string& option, VCLLogger* vclLogger,
                               std::unordered_map<std::string, T>& results, T (*function)(std::string, bool&)) {
    /// The content of option may like --inputs_precisions="A:fp16", the final key is A, value is
    /// InferenceEngine::Precision::FP16
    std::size_t firstDelimPos = option.find_first_of('"');
    std::size_t lastDelimPos = option.find_last_of('"');
    /// The stream may like A:FP32 B:FP32 C:U8
    std::istringstream stream(option.substr(firstDelimPos + 1, lastDelimPos - (firstDelimPos + 1)));
    std::string elem;
    /// Not all values are legal and can be converted to special type by function
    bool matched;
    /// Parse and save value for each element
    while (stream >> elem) {
        /// ':' is the seperator of element name and element value
        std::size_t lastDelimPos = elem.find_last_of(':');
        if (lastDelimPos == std::string::npos) {
            vclLogger->outputError(formatv("Failed to find delim in option! Value: {0}", elem));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        std::string key = elem.substr(0, lastDelimPos);
        std::string val = elem.substr(lastDelimPos + 1);
        vclLogger->debug("ioInfo options - key: {0} value: {1}", key, val);
        results[key] = function(val, matched);
        if (!matched) {
            /// Return error if the setting is not in list.
            /// Support "ANY" layout and "UNSPECIFIED" precision can increase robustness.
            vclLogger->outputError(formatv("Failed to find {0} for {1}", val, key));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }
    return VCL_RESULT_SUCCESS;
}

template <typename T>
inline void myTransform(T& value) {
}

/**
 * @brief Convert string content to upper case
 */
template <>
inline void myTransform<std::string>(std::string& value) {
    std::transform(value.begin(), value.end(), value.begin(), toupper);
}

/**
 * @brief Helper function to convert value to special type in container
 *
 * @tparam KEY The type of key in container
 * @tparam VAL The type of value in container
 * @param key The key that need to be converted to value in container
 * @param matched If we can find the key in container, it is matched
 * @param con The container of the KEY and VALUE we supported
 * @param defaultValue If we do not find the key in container, we will use defaultValue for the key
 * @return VAL The matched value of the key in container
 */
template <typename KEY, typename VAL>
VAL getElementFromCon(KEY& key, bool& matched, const std::unordered_map<KEY, VAL>& con, VAL defaultValue) {
    myTransform<KEY>(key);
    const auto elem = con.find(key);
    if (elem == con.end()) {
        // For unknown value, use default value.
        matched = false;
        return defaultValue;
    } else {
        matched = true;
        return elem->second;
    }
}

BuildInfo::BuildInfo(VPUXCompilerL0* pvc): pvc(pvc), parsedConfig(pvc->getOptions()), logger(pvc->getLogger()) {
}

InferenceEngine::Precision BuildInfo::getPrecisionIE(std::string value, bool& matched) {
    /// Remove some IE precisions to follow checkNetworkPrecision() of zero backend.
    /// Removed U64, I64, BF16, U16, I16, BOOL.
    /// @todo Update the map when zero backend begin to support more types
    static const std::unordered_map<std::string, InferenceEngine::Precision> supported_precisions = {
            {"FP32", InferenceEngine::Precision::FP32}, {"FP16", InferenceEngine::Precision::FP16},
            {"U32", InferenceEngine::Precision::U32},   {"I32", InferenceEngine::Precision::I32},
            {"U8", InferenceEngine::Precision::U8},     {"I8", InferenceEngine::Precision::I8},
    };

    return getElementFromCon<std::string, InferenceEngine::Precision>(value, matched, supported_precisions,
                                                                      InferenceEngine::Precision::UNSPECIFIED);
}

InferenceEngine::Layout BuildInfo::getLayoutIE(std::string value, bool& matched) {
    /// @todo Update map when the supported layout changed
    static const std::unordered_map<std::string, InferenceEngine::Layout> supported_layouts = {
            {"NCDHW", InferenceEngine::Layout::NCDHW}, {"NDHWC", InferenceEngine::Layout::NDHWC},
            {"NCHW", InferenceEngine::Layout::NCHW},   {"NHWC", InferenceEngine::Layout::NHWC},
            {"CHW", InferenceEngine::Layout::CHW},     {"HWC", InferenceEngine::Layout::HWC},
            {"NC", InferenceEngine::Layout::NC},       {"C", InferenceEngine::Layout::C},
    };

    return getElementFromCon<std::string, InferenceEngine::Layout>(value, matched, supported_layouts,
                                                                   InferenceEngine::Layout::ANY);
}

vcl_result_t BuildInfo::parseIOOption(const std::vector<std::string>& ioInfoOptions) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    /// Parse the precision && layout of input && output
    for (auto& option : ioInfoOptions) {
        if (option.find(KEY_INPUTS_PRECISIONS) != std::string::npos) {
            ret = parseSingleOption(option, logger, inPrcsIE, getPrecisionIE);
        } else if (option.find(KEY_INPUTS_LAYOUTS) != std::string::npos) {
            ret = parseSingleOption(option, logger, inLayoutsIE, getLayoutIE);
        } else if (option.find(KEY_OUTPUTS_PRECISIONS) != std::string::npos) {
            ret = parseSingleOption(option, logger, outPrcsIE, getPrecisionIE);
        } else if (option.find(KEY_OUTPUTS_LAYOUTS) != std::string::npos) {
            ret = parseSingleOption(option, logger, outLayoutsIE, getLayoutIE);
        } else {
            logger->outputError(formatv("Invalid key in option! Option: {0}", option));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (ret != VCL_RESULT_SUCCESS)
            return ret;
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t BuildInfo::prepareBuildFlags(const std::string& descOptions) {
    /// Find the location of special separator in descOptions, the separator helps us to find input options, output
    /// options, config options
    std::size_t inputPrecisionSeparator = descOptions.find(KEY_INPUTS_PRECISIONS);
    std::size_t inputLayoutSeparator = descOptions.find(KEY_INPUTS_LAYOUTS);
    std::size_t inputModelLayoutSeparator = descOptions.find(KEY_INPUTS_MODEL_LAYOUTS);
    std::size_t outputPrecisionSeparator = descOptions.find(KEY_OUTPUTS_PRECISIONS);
    std::size_t outputLayoutSeparator = descOptions.find(KEY_OUTPUTS_LAYOUTS);
    std::size_t outputModelLayoutSeparator = descOptions.find(KEY_OUTPUTS_MODEL_LAYOUTS);
    std::size_t configSeparator = descOptions.find(KEY_CONFIGS);

    /// Parse the options for input && output
    std::vector<std::string> ioInfoOptions;
    if (inputPrecisionSeparator != std::string::npos && inputLayoutSeparator != std::string::npos &&
        outputPrecisionSeparator != std::string::npos && outputLayoutSeparator != std::string::npos) {
        /// Separate ioInfo to different section
        ioInfoOptions.push_back(descOptions.substr(inputPrecisionSeparator, inputLayoutSeparator));
        if (inputModelLayoutSeparator != std::string::npos) {
            ioInfoOptions.push_back(
                    descOptions.substr(inputLayoutSeparator, inputModelLayoutSeparator - inputLayoutSeparator));
            ioInfoOptions.push_back(descOptions.substr(inputModelLayoutSeparator,
                                                       outputPrecisionSeparator - inputModelLayoutSeparator));
        } else {
            ioInfoOptions.push_back(
                    descOptions.substr(inputLayoutSeparator, outputPrecisionSeparator - inputLayoutSeparator));
        }
        ioInfoOptions.push_back(
                descOptions.substr(outputPrecisionSeparator, outputLayoutSeparator - outputPrecisionSeparator));
        if (configSeparator != std::string::npos) {
            if (outputModelLayoutSeparator != std::string::npos) {
                ioInfoOptions.push_back(
                        descOptions.substr(outputLayoutSeparator, outputModelLayoutSeparator - outputLayoutSeparator));
                ioInfoOptions.push_back(
                        descOptions.substr(outputModelLayoutSeparator, configSeparator - outputModelLayoutSeparator));
            } else {
                ioInfoOptions.push_back(
                        descOptions.substr(outputLayoutSeparator, configSeparator - outputLayoutSeparator));
            }
        } else {
            if (outputModelLayoutSeparator != std::string::npos) {
                ioInfoOptions.push_back(
                        descOptions.substr(outputLayoutSeparator, outputModelLayoutSeparator - outputLayoutSeparator));
                ioInfoOptions.push_back(descOptions.substr(outputModelLayoutSeparator));
            } else {
                ioInfoOptions.push_back(descOptions.substr(outputLayoutSeparator));
            }
        }
    } else {
        /// Return error if the mandatory ioInfo options are not passed
        /// Skip ioInfo missing if is used for debug.
        bool skipIOInfo = false;
#if defined(VPUX_DEVELOPER_BUILD)
        if (const auto env = std::getenv("IE_VPUX_VCL_SKIP_IOINFO")) {
            skipIOInfo = std::stoi(env);
        }
#endif
        if (skipIOInfo == false) {
            logger->outputError(formatv("Mandatory ioInfo options are missing! DescOptions: {0}", descOptions));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }

    /// Parse the compilation options
    std::vector<std::string> options;
    if (configSeparator != std::string::npos) {
        /// Skip "--config" during parsing, The content may like:
        /// NPU_PLATFORM="3720"  NPU_COMPILATION_MODE_PARAMS="swap-transpose-with-fq=1 force-z-major-concat=1"
        std::string content = descOptions.substr(configSeparator + strlen(KEY_CONFIGS));
        // From 5.0.0, compiler only support NPU_ prefix, replace VPUX_ or VPU_ with NPU_
        std::regex reg("VPUX_");
        content = std::regex_replace(content, reg, "NPU_");
        reg = "VPU_";
        content = std::regex_replace(content, reg, "NPU_");
        std::stringstream input(content);

        /// A singleOption is consist of one or more words, like value of VPUX_COMPILATION_MODE_PARAMS
        std::string word;
        std::string singleOption = "";
        while (input >> word) {
            if (singleOption.compare("") == 0) {
                /// Save the first word
                singleOption = word;
            } else {
                /// Save the word that belongs to this option
                singleOption = singleOption + " " + word;
            }

            /// If this is not the last word of this option, contine for this option
            if (word[word.size() - 1] != '"') {
                continue;
            }

            /// Save current option
            options.push_back(singleOption);
            /// Clean for the next option
            singleOption = "";
        }
        /// Save the last option
        if (singleOption.compare("") != 0) {
            options.push_back(singleOption);
        }
    }

    /// Show all parsed options
    for (auto& op : options) {
        logger->debug("option : {0}", op);
    }

    /// Save the parsed configs from user
    std::map<std::string, std::string> config;

    /// Pase compilation options and save to config
    /// User options will overwrite default values in config
    try {
        for (auto& option : options) {
            if (option.find_first_not_of(' ') == std::string::npos) {
                continue;
            }
            size_t length = option.size();
            /// Skip the terminator of string
            if (option[length - 1] == '\0') {
                length--;
            }

            std::size_t lastDelimPos = option.find_first_of('=');
            /// Use 2 to skip =" , the format shall follow key="value"
            if (lastDelimPos == std::string::npos || lastDelimPos + 2 > length) {
                throw std::logic_error(option + " is in bad format!");
            }
            std::string key = option.substr(0, lastDelimPos);
            /// For key="value", the val shall be value
            /// Skip =" in the front and " at the end
            config[key] = option.substr(lastDelimPos + 2, length - 1 - (lastDelimPos + 2));
            logger->debug("config options - key: {0} value: {1}", key, config[key]);
        }
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError("Internal exception in config parser!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// Foce to use MLIR compiler.
    config[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);

    // Use platform information provided by driver if platform config is either not found or set on AUTO_DETECT
    if (config.find(VPUX_CONFIG_KEY(PLATFORM)) == config.end() || "AUTO_DETECT" == config[VPUX_CONFIG_KEY(PLATFORM)]) {
        // Set platform
        switch (pvc->getCompilerDesc().platform) {
        case VCL_PLATFORM_VPU3700:
            config[VPUX_CONFIG_KEY(PLATFORM)] = "3700";
            config[CONFIG_KEY(DEVICE_ID)] = "3700";
            break;
        case VCL_PLATFORM_VPU3720:
            config[VPUX_CONFIG_KEY(PLATFORM)] = "3720";
            config[CONFIG_KEY(DEVICE_ID)] = "3720";
            break;
        default:
            logger->outputError(formatv("Unrecognized platform! {0}", pvc->getCompilerDesc().platform));
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        };
    }

    /// When we use LOG_INFO, show vcl level profiling log
    std::map<std::string, std::string>::iterator iter = config.find(CONFIG_KEY(LOG_LEVEL));
    if (iter != config.end()) {
        if (iter->second == CONFIG_VALUE(LOG_INFO))
            enableProfiling = true;
    }

    /// Update default compilation config options with the new values we parsed from user descriptions
    try {
        parsedConfig.update(config, OptionMode::CompileTime);
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError(formatv("Internal exception! Can not update config! DescOptions: {0}", descOptions));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// If user sepecify preferred log level, update our logger
    if (iter != config.end()) {
        logger->setLevel(parsedConfig.get<LOG_LEVEL>());
    }

    /// Show compiler ID which helps to find the commit of compiler
    logger->info("Current driver compiler ID: {0}", pvc->getCompilerProp().id);
    logger->info("Current build flags: {0}", descOptions);

    /// Parse precision and layout info of input && output from user
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    try {
        ret = parseIOOption(ioInfoOptions);
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        logger->outputError(formatv("Internal exception! Can't parse ioInfo! DescOptions: {0}", descOptions));
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (ret != VCL_RESULT_SUCCESS) {
        logger->outputError(formatv("Failed to parse ioInfoOptions! DescOptions: {0}", descOptions));
        return ret;
    }
    return ret;
}

vcl_result_t BuildInfo::prepareModel(const uint8_t* modelIR, uint64_t modelIRSize) {
    if (modelIR == nullptr) {
        logger->outputError("Invalid model ir pointer!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// The API version of current compiler, adapter fill its version in modelIR, shall be same value
    vcl_version_info_t currentAPIVersion = pvc->getCompilerProp().version;
    uint32_t offset = 0;
    vcl_version_info_t APIVersion;
    memcpy(&APIVersion, modelIR, sizeof(APIVersion));
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        logger->outputError(formatv("Unsupported IR API version! Val: {0}.{1}", APIVersion.major, APIVersion.minor));
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(vcl_version_info_t);

    /// The number of elements in buffer shall not exceed limitation
    uint32_t numOfElements = 0;
    memcpy(&numOfElements, modelIR + offset, sizeof(numOfElements));
    if (numOfElements >= maxNumberOfElements) {
        logger->outputError("Bad elements number in IR!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(numOfElements);

    /// The size of model data
    uint64_t bufferSize = 0;
    memcpy(&bufferSize, modelIR + offset, sizeof(bufferSize));
    if (bufferSize == 0 || bufferSize >= maxSizeOfXML) {
        logger->outputError("Bad buffer size in IR!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(bufferSize);

    /// The offset to model xml
    uint64_t bufferOffset = offset;
    offset += bufferSize;

    /// The size of model weight
    uint64_t weightsSize = 0;
    memcpy(&weightsSize, modelIR + offset, sizeof(weightsSize));
    if (weightsSize >= maxSizeOfWeights) {
        logger->outputError("Bad weights size in IR!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(weightsSize);

    /// The offset to model weight
    uint64_t weightsOffset = offset;
    if (offset + weightsSize > modelIRSize) {
        logger->outputError("The IR content and size mismatch!");
        return VCL_RESULT_ERROR_INVALID_IR;
    }

    /// The pointer to model xml
    const uint8_t* buffer = modelIR + bufferOffset;
    /// The pointer to model weight
    const uint8_t* weights = modelIR + weightsOffset;
    /// Create CNNNetwork with model data
    try {
        std::string modelData(buffer, buffer + bufferSize);
        ov::runtime::Tensor weightsTensor;
        if (weightsSize > 0)
            weightsTensor = ov::runtime::Tensor(ov::element::u8, {weightsSize}, const_cast<uint8_t*>(weights));
        ov::Core core;

        StopWatch stopWatch;
        if (enableProfiling) {
            stopWatch.start();
        }

        std::shared_ptr<ov::Model> model = core.read_model(modelData, weightsTensor);
        /// Current compiler still uses CNNNetwork
        /// @todo Use OV::Model once we drop CNNNetwork
        // cnnNet = InferenceEngine::CNNNetwork(std::const_pointer_cast<ngraph::Function>(model));
        cnnNet = InferenceEngine::CNNNetwork(model);

        if (enableProfiling) {
            stopWatch.stop();
            logger->info("The time to convert data to model: {0} ms", stopWatch.delta_ms());
        }
    } catch (const std::exception& error) {
        logger->outputError(error.what());
        return VCL_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        logger->outputError("Internal exception! Can not create cnnnetwork!");
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    return VCL_RESULT_SUCCESS;
}

}  // namespace VPUXDriverCompiler

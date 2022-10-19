//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "VPUXCompilerL0.h"

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>
#include <openvino/openvino.hpp>

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"
#include "vpux/vpux_plugin_config.hpp"
#include "vpux_compiler.hpp"
#include "vpux_private_config.hpp"

#include <chrono>
#include <istream>
#include <sstream>
#include <string>
#include <utility>

#define xstr(s) str(s)
#define str(s) #s

#define COMPILER_MAJOR 2
#define COMPILER_MINOR 1
static const char* COMPILER_VERSION = xstr(COMPILER_MAJOR) "." xstr(COMPILER_MINOR);

#define PROFILING_MAJOR 1
#define PROFILING_MINOR 0

#define KEY_INPUTS_PRECISIONS "--inputs_precisions"
#define KEY_INPUTS_LAYOUTS "--inputs_layouts"
#define KEY_INPUTS_MODEL_LAYOUTS "--inputs_model_layouts"
#define KEY_OUTPUTS_PRECISIONS "--outputs_precisions"
#define KEY_OUTPUTS_LAYOUTS "--outputs_layouts"
#define KEY_OUTPUTS_MODEL_LAYOUTS "--outputs_model_layouts"
#define KEY_CONFIGS "--config"
#define KEY_VCL_OV_API_2 "VCL_OV_API_2"

const uint32_t maxNumberOfElements = 10;
const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

using namespace vpux;

namespace VPUXCompilerL0 {

template <typename T>
vcl_result_t parseSingleOption(std::string& option, uint32_t debugLevel, std::unordered_map<std::string, T>& arrays,
                               T (*function)(std::string, bool&)) {
    // The ioInfo may like --inputs_precisions="A:fp16", the stream shall be A:fp16
    std::size_t firstDelimPos = option.find_first_of('"');
    std::size_t lastDelimPos = option.find_last_of('"');
    std::istringstream stream(option.substr(firstDelimPos + 1, lastDelimPos - (firstDelimPos + 1)));
    std::string elem;
    bool matched;
    while (stream >> elem) {
        // The stream may like A:fp16
        std::size_t lastDelimPos = elem.find_last_of(':');
        if (lastDelimPos == std::string::npos) {
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
        std::string key = elem.substr(0, lastDelimPos);
        std::string val = elem.substr(lastDelimPos + 1);
        if (debugLevel > 3) {
            std::cout << "ioInfo options - key: " << key << " value: " << val << std::endl;
        }
        arrays[key] = function(val, matched);
        if (!matched) {
            // Return error if the setting is not in list.
            // Support "ANY" layout and "UNSPECIFIED" precision can increase robustness.
            if (debugLevel > 3)
                std::cout << "Failed to find " << val << " for " << key << std::endl;
            return VCL_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }
    return VCL_RESULT_SUCCESS;
}

template <typename T>
inline void myTransform(T& value) {
}

template <>
inline void myTransform<std::string>(std::string& value) {
    std::transform(value.begin(), value.end(), value.begin(), toupper);
}

template <typename KEY, typename VAL>
VAL getElementFromCon(KEY key, bool& matched, const std::unordered_map<KEY, VAL>& con, VAL defaultValue) {
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

struct IOInfoV1 {
    static InferenceEngine::Precision getPrecisionIE(std::string value, bool& matched) {
        // Remove some IE precisions to follow checkNetworkPrecision().
        // Removed U64, I64, BF16, U16, I16, BOOL.
        static const std::unordered_map<std::string, InferenceEngine::Precision> supported_precisions = {
                {"FP32", InferenceEngine::Precision::FP32}, {"FP16", InferenceEngine::Precision::FP16},
                {"U32", InferenceEngine::Precision::U32},   {"I32", InferenceEngine::Precision::I32},
                {"U8", InferenceEngine::Precision::U8},     {"I8", InferenceEngine::Precision::I8},
        };

        return getElementFromCon<std::string, InferenceEngine::Precision>(value, matched, supported_precisions,
                                                                          InferenceEngine::Precision::UNSPECIFIED);
    }

    static InferenceEngine::Layout getLayoutIE(std::string value, bool& matched) {
        static const std::unordered_map<std::string, InferenceEngine::Layout> supported_layouts = {
                {"NCDHW", InferenceEngine::Layout::NCDHW}, {"NDHWC", InferenceEngine::Layout::NDHWC},
                {"NCHW", InferenceEngine::Layout::NCHW},   {"NHWC", InferenceEngine::Layout::NHWC},
                {"CHW", InferenceEngine::Layout::CHW},     {"HWC", InferenceEngine::Layout::HWC},
                {"NC", InferenceEngine::Layout::NC},       {"C", InferenceEngine::Layout::C},
        };

        return getElementFromCon<std::string, InferenceEngine::Layout>(value, matched, supported_layouts,
                                                                       InferenceEngine::Layout::ANY);
    }

    vcl_result_t parse(std::vector<std::string>& ioInfoOptions, uint32_t debugLevel) {
        vcl_result_t ret = VCL_RESULT_SUCCESS;
        for (auto& option : ioInfoOptions) {
            if (option.find(KEY_INPUTS_PRECISIONS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, inPrcsIE, getPrecisionIE);
            } else if (option.find(KEY_INPUTS_LAYOUTS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, inLayoutsIE, getLayoutIE);
            } else if (option.find(KEY_OUTPUTS_PRECISIONS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, outPrcsIE, getPrecisionIE);
            } else if (option.find(KEY_OUTPUTS_LAYOUTS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, outLayoutsIE, getLayoutIE);
            } else {
                return VCL_RESULT_ERROR_INVALID_ARGUMENT;
            }
            if (ret != VCL_RESULT_SUCCESS)
                return ret;
        }
        return VCL_RESULT_SUCCESS;
    }

    // OV 1.0
    std::unordered_map<std::string, InferenceEngine::Precision> inPrcsIE;
    std::unordered_map<std::string, InferenceEngine::Layout> inLayoutsIE;
    std::unordered_map<std::string, InferenceEngine::Precision> outPrcsIE;
    std::unordered_map<std::string, InferenceEngine::Layout> outLayoutsIE;
};  // IOInfoV1

struct IOInfoV2 {
    static ov::element::Type getPrecisionOV(std::string value, bool& matched) {
        // Remove some IE precisions to follow checkNetworkPrecision().
        // Removed U64, I64, BF16, U16, I16, BOOL.
        static const std::unordered_map<std::string, ov::element::Type> supported_precisions = {
                {"FP32", ov::element::f32}, {"FP16", ov::element::f16}, {"U32", ov::element::u32},
                {"I32", ov::element::i32},  {"U8", ov::element::u8},    {"I8", ov::element::i8},
        };

        return getElementFromCon<std::string, ov::element::Type>(value, matched, supported_precisions,
                                                                 ov::element::undefined);
    }

    static ov::Layout getLayoutOV(std::string value, bool& matched) {
        static const std::unordered_map<std::string, ov::Layout> supported_layouts = {
                {"NCDHW", ov::Layout("NCDHW")}, {"NDHWC", ov::Layout("NDHWC")}, {"NCHW", ov::Layout("NCHW")},
                {"NHWC", ov::Layout("NHWC")},   {"CHW", ov::Layout("CHW")},     {"HWC", ov::Layout("HWC")},
                {"NC", ov::Layout("NC")},       {"C", ov::Layout("C")},
        };
        return getElementFromCon<std::string, ov::Layout>(value, matched, supported_layouts, ov::Layout());
    }

    static InferenceEngine::Precision convertOVPrecisionToIEPrecision(const ov::element::Type value, bool& matched) {
        static const std::unordered_map<ov::element::Type_t, InferenceEngine::Precision> supported_precisions = {
                {ov::element::Type_t::f32, InferenceEngine::Precision::FP32},
                {ov::element::Type_t::f16, InferenceEngine::Precision::FP16},
                {ov::element::Type_t::u32, InferenceEngine::Precision::U32},
                {ov::element::Type_t::i32, InferenceEngine::Precision::I32},
                {ov::element::Type_t::u8, InferenceEngine::Precision::U8},
                {ov::element::Type_t::i8, InferenceEngine::Precision::I8},
        };

        return getElementFromCon<ov::element::Type_t, InferenceEngine::Precision>(
                ov::element::Type_t(value), matched, supported_precisions, InferenceEngine::Precision::UNSPECIFIED);
    }

    static InferenceEngine::Layout convertOVLayoutToIELayout(std::string value, bool& matched) {
        static const std::unordered_map<std::string, InferenceEngine::Layout> supported_layouts = {
                {"[N,C,D,H,W]", InferenceEngine::Layout::NCDHW}, {"[N,D,H,W,C]", InferenceEngine::Layout::NDHWC},
                {"[N,C,H,W]", InferenceEngine::Layout::NCHW},    {"[N,H,W,C]", InferenceEngine::Layout::NHWC},
                {"[C,H,W]", InferenceEngine::Layout::CHW},       {"[H,W,C]", InferenceEngine::Layout::HWC},
                {"[N,C]", InferenceEngine::Layout::NC},          {"[C]", InferenceEngine::Layout::C},
        };
        return getElementFromCon<std::string, InferenceEngine::Layout>(value, matched, supported_layouts,
                                                                       InferenceEngine::Layout::ANY);
    }

    vcl_result_t parse(std::vector<std::string>& ioInfoOptions, uint32_t debugLevel) {
        vcl_result_t ret = VCL_RESULT_SUCCESS;
        for (auto& option : ioInfoOptions) {
            if (option.find(KEY_INPUTS_PRECISIONS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, inPrcsOV, getPrecisionOV);
            } else if (option.find(KEY_INPUTS_LAYOUTS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, inLayoutsOV, getLayoutOV);
            } else if (option.find(KEY_INPUTS_MODEL_LAYOUTS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, inMLayoutsOV, getLayoutOV);
            } else if (option.find(KEY_OUTPUTS_PRECISIONS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, outPrcsOV, getPrecisionOV);
            } else if (option.find(KEY_OUTPUTS_LAYOUTS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, outLayoutsOV, getLayoutOV);
            } else if (option.find(KEY_OUTPUTS_MODEL_LAYOUTS) != std::string::npos) {
                ret = parseSingleOption(option, debugLevel, outMLayoutsOV, getLayoutOV);
            } else {
                return VCL_RESULT_ERROR_INVALID_ARGUMENT;
            }
            if (ret != VCL_RESULT_SUCCESS)
                return ret;
        }
        return VCL_RESULT_SUCCESS;
    }

    // OV 2.0
    std::unordered_map<std::string, ov::element::Type> inPrcsOV;
    std::unordered_map<std::string, ov::Layout> inLayoutsOV;
    std::unordered_map<std::string, ov::Layout> inMLayoutsOV;
    std::unordered_map<std::string, ov::element::Type> outPrcsOV;
    std::unordered_map<std::string, ov::Layout> outLayoutsOV;
    std::unordered_map<std::string, ov::Layout> outMLayoutsOV;
};  // IOInfoV2

struct StopWatch {
    using fp_milliseconds = std::chrono::duration<double, std::chrono::milliseconds::period>;

    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    void stop() {
        stopTime = std::chrono::steady_clock::now();
    }

    double delta_ms() {
        return std::chrono::duration_cast<fp_milliseconds>(stopTime - startTime).count();
    }

    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point stopTime;
};

class VPUXExecutableL0 final {
public:
    VPUXExecutableL0(NetworkDescription::Ptr networkDesc, bool enableProfiling);
    vcl_result_t serializeNetwork();
    vcl_result_t getNetworkSize(uint64_t* blobSize);
    vcl_result_t exportNetwork(uint8_t* blob, uint64_t blobSize);

private:
    NetworkDescription::Ptr _networkDesc;
    bool enableProfiling;
    std::vector<char> _blob;
};

VPUXExecutableL0::VPUXExecutableL0(NetworkDescription::Ptr networkDesc, bool enableProfiling)
        : _networkDesc(networkDesc), enableProfiling(enableProfiling) {
    _blob.clear();
}

vcl_result_t VPUXExecutableL0::serializeNetwork() {
    StopWatch stopWatch;
    if (enableProfiling)
        stopWatch.start();

    _blob = _networkDesc->getCompiledNetwork();

    if (enableProfiling) {
        stopWatch.stop();
        std::cout << "getCompiledNetwork time: " << stopWatch.delta_ms() << "ms" << std::endl;
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXExecutableL0::getNetworkSize(uint64_t* blobSize) {
    if (blobSize == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    *blobSize = _blob.size();
    if (*blobSize == 0) {
        // The executable handle do not contain a legal network.
        return VCL_RESULT_ERROR_UNKNOWN;
    } else {
        return VCL_RESULT_SUCCESS;
    }
}

vcl_result_t VPUXExecutableL0::exportNetwork(uint8_t* blob, uint64_t blobSize) {
    if (!blob || blobSize != _blob.size()) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    StopWatch stopWatch;
    if (enableProfiling)
        stopWatch.start();

    memcpy(blob, _blob.data(), blobSize);

    if (enableProfiling) {
        stopWatch.stop();
        std::cout << "exportNetwork time: " << stopWatch.delta_ms() << "ms" << std::endl;
    }
    return VCL_RESULT_SUCCESS;
}

class VPUXCompilerL0 final {
public:
    VPUXCompilerL0(vcl_compiler_desc_t desc, std::map<std::string, std::string>& config);

    vcl_compiler_properties_t getCompilerProp() const {
        return _compilerProp;
    }
    vcl_compiler_desc_t getCompilerDesc() const {
        return _compilerDesc;
    }

    std::shared_ptr<const OptionsDesc> getOptions() const {
        return _options;
    }

    std::pair<VPUXExecutableL0*, vcl_result_t> importNetworkV1(const uint8_t* buffer, uint64_t bufferSize,
                                                               const uint8_t* weights, uint64_t weightsSize,
                                                               Config& vpuxConfig, const IOInfoV1& ioInfo,
                                                               bool enableProfiling);

    std::pair<VPUXExecutableL0*, vcl_result_t> importNetworkV2(const uint8_t* buffer, uint64_t bufferSize,
                                                               const uint8_t* weights, uint64_t weightsSize,
                                                               Config& vpuxConfig, const IOInfoV2& ioInfo,
                                                               bool enableProfiling);

private:
    std::shared_ptr<OptionsDesc> _options;
    Compiler::Ptr _compiler = NULL;
    vcl_compiler_properties_t _compilerProp;
    vcl_compiler_desc_t _compilerDesc;
    std::mutex _mlock;
};

VPUXCompilerL0::VPUXCompilerL0(vcl_compiler_desc_t desc, std::map<std::string, std::string>& config)
        : _options(std::make_shared<OptionsDesc>()) {
    registerCommonOptions(*_options);
    registerCompilerOptions(*_options);
    registerRunTimeOptions(*_options);

    Config parsedConfig(_options);
    parsedConfig.update(config, OptionMode::CompileTime);
    _compiler = Compiler::create(parsedConfig);

    _compilerDesc = desc;
    _compilerProp.id = COMPILER_VERSION;
    _compilerProp.version.major = COMPILER_MAJOR;
    _compilerProp.version.minor = COMPILER_MINOR;
    _compilerProp.supportedOpsets = 7;
}

std::pair<VPUXExecutableL0*, vcl_result_t> VPUXCompilerL0::importNetworkV1(const uint8_t* buffer, uint64_t bufferSize,
                                                                           const uint8_t* weights, uint64_t weightsSize,
                                                                           Config& config, const IOInfoV1& ioInfo,
                                                                           bool enableProfiling) {
    if (buffer == nullptr || weights == nullptr) {
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }
    std::string model(buffer, buffer + bufferSize);
    InferenceEngine::MemoryBlob::Ptr weightsBlob;
    if (weightsSize != 0) {
        InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {weightsSize},
                                               InferenceEngine::Layout::C);
        weightsBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, const_cast<uint8_t*>(weights));
    }

    InferenceEngine::Core ieCore;
    StopWatch stopWatch;
    if (enableProfiling)
        stopWatch.start();
    InferenceEngine::CNNNetwork cnnNet = ieCore.ReadNetwork(model, weightsBlob);
    if (enableProfiling) {
        stopWatch.stop();
        std::cout << "ReadNetwork time: " << stopWatch.delta_ms() << "ms" << std::endl;
        stopWatch.start();
    }

    NetworkDescription::Ptr networkDesc = NULL;
    try {
        // Update input and output info
        auto inputs = cnnNet.getInputsInfo();
        auto outputs = cnnNet.getOutputsInfo();

        for (const auto& item : ioInfo.inPrcsIE) {
            const auto& name = item.first;
            const auto input = inputs.find(name);
            if (input != inputs.end()) {
                input->second->setPrecision(item.second);
            } else {
                throw std::logic_error(name + " is not found in inputs to set precision!");
            }
        }

        for (const auto& item : ioInfo.inLayoutsIE) {
            const auto& name = item.first;
            const auto input = inputs.find(name);
            if (input != inputs.end()) {
                input->second->setLayout(item.second);
            } else {
                throw std::logic_error(name + " is not found in inputs to set layout!");
            }
        }

        for (const auto& item : ioInfo.outPrcsIE) {
            const auto& name = item.first;
            const auto output = outputs.find(name);
            if (output != outputs.end()) {
                output->second->setPrecision(item.second);
            } else {
                throw std::logic_error(name + " is not found in outputs to set precision!");
            }
        }

        for (const auto& item : ioInfo.outLayoutsIE) {
            const auto& name = item.first;
            const auto output = outputs.find(name);
            if (output != outputs.end()) {
                output->second->setLayout(item.second);
            } else {
                throw std::logic_error(name + " is not found in outputs to set layout!");
            }
        }

        networkDesc = _compiler->compile(cnnNet.getFunction(), cnnNet.getName(), inputs, outputs, config);
    } catch (const std::exception& error) {
        if (_compilerDesc.debug_level > 0)
            std::cerr << error.what() << std::endl;
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    } catch (...) {
        if (_compilerDesc.debug_level > 0)
            std::cerr << "Internal exception!" << std::endl;
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }
    if (enableProfiling) {
        stopWatch.stop();
        std::cout << "Compile net time: " << stopWatch.delta_ms() << "ms" << std::endl;
    }
    VPUXExecutableL0* exe = new VPUXExecutableL0(networkDesc, enableProfiling);
    return std::pair<VPUXExecutableL0*, vcl_result_t>(exe, VCL_RESULT_SUCCESS);
}

std::pair<VPUXExecutableL0*, vcl_result_t> VPUXCompilerL0::importNetworkV2(const uint8_t* buffer, uint64_t bufferSize,
                                                                           const uint8_t* weights, uint64_t weightsSize,
                                                                           Config& config, const IOInfoV2& ioInfo,
                                                                           bool enableProfiling) {
    if (buffer == nullptr || weights == nullptr) {
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }

    std::string modelData(buffer, buffer + bufferSize);
    ov::runtime::Tensor weightsTensor;
    if (weightsSize > 0)
        weightsTensor = ov::runtime::Tensor(ov::element::u8, {weightsSize}, const_cast<uint8_t*>(weights));
    ov::Core core;
    StopWatch stopWatch;
    if (enableProfiling)
        stopWatch.start();
    auto model = core.read_model(modelData, weightsTensor);
    if (enableProfiling) {
        stopWatch.stop();
        std::cout << "ReadNetwork time: " << stopWatch.delta_ms() << "ms" << std::endl;
        stopWatch.start();
    }

    NetworkDescription::Ptr networkDesc = NULL;
    try {
        auto preprocessor = ov::preprocess::PrePostProcessor(model);
        const auto inputs = model->inputs();
        const auto outputs = model->outputs();

        // Update input and output info
        for (auto&& item : ioInfo.inPrcsOV) {
            const auto& name = item.first;
            bool found = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (!inputs[i].get_node()->get_friendly_name().compare(name)) {
                    preprocessor.input(i).tensor().set_element_type(item.second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to find " << name << " in inputs to set precision!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
        }

        for (auto&& item : ioInfo.inLayoutsOV) {
            const auto& name = item.first;
            bool found = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (!inputs[i].get_node()->get_friendly_name().compare(name)) {
                    preprocessor.input(i).tensor().set_layout(item.second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to find " << name << " in inputs to set layout!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
        }

        for (auto&& item : ioInfo.inMLayoutsOV) {
            const auto& name = item.first;
            bool found = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (!inputs[i].get_node()->get_friendly_name().compare(name)) {
                    preprocessor.input(i).model().set_layout(item.second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to find " << name << " in inputs to set model layout!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
        }

        for (auto&& item : ioInfo.outPrcsOV) {
            const auto& name = item.first;
            bool found = false;
            for (size_t i = 0; i < outputs.size(); i++) {
                if (!outputs[i].get_node()->get_input_node_ptr(0)->get_friendly_name().compare(name)) {
                    preprocessor.output(i).tensor().set_element_type(item.second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to find " << name << " in outputs to set precision!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
        }

        for (auto&& item : ioInfo.outLayoutsOV) {
            const auto& name = item.first;
            bool found = false;
            for (size_t i = 0; i < outputs.size(); i++) {
                if (!outputs[i].get_node()->get_input_node_ptr(0)->get_friendly_name().compare(name)) {
                    preprocessor.output(i).tensor().set_layout(item.second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to find " << name << " in outputs to set layout!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
        }

        for (auto&& item : ioInfo.outMLayoutsOV) {
            const auto& name = item.first;
            bool found = false;
            for (size_t i = 0; i < outputs.size(); i++) {
                if (!outputs[i].get_node()->get_input_node_ptr(0)->get_friendly_name().compare(name)) {
                    preprocessor.output(i).model().set_layout(item.second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to find " << name << " in outputs to set model layout!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
        }

        model = preprocessor.build();
        // Create inputsInfo
        // As model layout can impact the info, need the read from processed model.
        bool matched = false;
        InferenceEngine::InputsDataMap inputsInfo;
        for (auto&& input : model->inputs()) {
            std::string name = input.get_node()->get_friendly_name();
            InferenceEngine::InputInfo::Ptr ieInfo = std::make_shared<InferenceEngine::InputInfo>();
            InferenceEngine::Layout layout =
                    IOInfoV2::convertOVLayoutToIELayout(ov::layout::get_layout(input).to_string(), matched);
            if (!matched) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to convert ov layout " << ov::layout::get_layout(input).to_string()
                              << " to IE layout!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
            auto precision = IOInfoV2::convertOVPrecisionToIEPrecision(input.get_element_type(), matched);
            if (!matched) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to convert ov precision to IE precision!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
            InferenceEngine::DataPtr data = std::make_shared<InferenceEngine::Data>(name, precision, layout);
            ieInfo->setInputData(data);
            inputsInfo[name] = ieInfo;
        }
        // Create outputsInfo
        InferenceEngine::OutputsDataMap outputsInfo;
        for (auto&& output : model->outputs()) {
            std::string name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
            InferenceEngine::Layout layout =
                    IOInfoV2::convertOVLayoutToIELayout(ov::layout::get_layout(output).to_string(), matched);
            if (!matched) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to convert ov layout " << ov::layout::get_layout(output).to_string()
                              << " to IE layout!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
            auto precision = IOInfoV2::convertOVPrecisionToIEPrecision(output.get_element_type(), matched);
            if (!matched) {
                if (_compilerDesc.debug_level > 0)
                    std::cerr << "Failed to convert ov precision to IE precision!" << std::endl;
                return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
            }
            InferenceEngine::DataPtr data = std::make_shared<InferenceEngine::Data>(name, precision, layout);
            outputsInfo[name] = data;
        }
        networkDesc = _compiler->compile(model, model->get_friendly_name(), inputsInfo, outputsInfo, config);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    } catch (...) {
        std::cerr << "Internal exception!" << std::endl;
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }
    if (enableProfiling) {
        stopWatch.stop();
        std::cout << "Compile net time: " << stopWatch.delta_ms() << "ms" << std::endl;
    }
    VPUXExecutableL0* exe = new VPUXExecutableL0(networkDesc, enableProfiling);
    return std::pair<VPUXExecutableL0*, vcl_result_t>(exe, VCL_RESULT_SUCCESS);
}

class VPUXProfilingL0 final {
public:
    VPUXProfilingL0(p_vcl_profiling_input_t profInput)
            : _blobData(profInput->blobData),
              _blobSize(profInput->blobSize),
              _profData(profInput->profData),
              _profSize(profInput->profSize) {
    }
    vcl_result_t preprocess();
    vcl_result_t getTaskInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getLayerInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getRawInfo(p_vcl_profiling_output_t profOutput);
    vcl_profiling_properties_t getProperties();

private:
    const uint8_t* _blobData;  ///< Pointer to the buffer with the blob
    uint64_t _blobSize;        ///< Size of the blob in bytes
    const uint8_t* _profData;  ///< Pointer to the raw profiling output
    uint64_t _profSize;        ///< Size of the raw profiling output

    std::vector<profiling::TaskInfo> _taskInfo;    ///< Per-task (DPU, DMA, SW) profiling info
    std::vector<profiling::LayerInfo> _layerInfo;  ///< Per-layer profiling info
};

vcl_result_t VPUXProfilingL0::preprocess() {
    auto result = VCL_RESULT_SUCCESS;
    try {
        _taskInfo = profiling::getTaskInfo(_blobData, _blobSize, _profData, _profSize, profiling::TaskType::ALL,
                                           profiling::VerbosityLevel::HIGH);
        _layerInfo = profiling::getLayerInfo(_taskInfo);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        result = VCL_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        std::cerr << "Internal exception! Can't parse profiling information." << std::endl;
        result = VCL_RESULT_ERROR_UNKNOWN;
    }
    return result;
}

vcl_result_t VPUXProfilingL0::getTaskInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_taskInfo.empty()) {
        std::cerr << "There are no tasks to return. Either profiling output was empty or it is not preprocessed."
                  << std::endl;
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    profOutput->data = reinterpret_cast<uint8_t*>(_taskInfo.data());
    profOutput->size = _taskInfo.size() * sizeof(profiling::TaskInfo);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXProfilingL0::getLayerInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_layerInfo.empty()) {
        std::cerr << "There are no layers to return. Either profiling output was empty or it is not preprocessed."
                  << std::endl;
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    profOutput->data = reinterpret_cast<uint8_t*>(_layerInfo.data());
    profOutput->size = _layerInfo.size() * sizeof(profiling::LayerInfo);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXProfilingL0::getRawInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    profOutput->data = _profData;
    profOutput->size = _profSize;
    return VCL_RESULT_SUCCESS;
}

vcl_profiling_properties_t VPUXProfilingL0::getProperties() {
    vcl_profiling_properties_t prop;
    prop.version.major = PROFILING_MAJOR;
    prop.version.minor = PROFILING_MINOR;
    return prop;
}

}  // namespace VPUXCompilerL0

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT vcl_result_t vclCompilerCreate(vcl_compiler_desc_t desc, vcl_compiler_handle_t* compiler) {
    // TODO: Check desc here once we limit the platform scope
    if (compiler == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // Set all default configs here
    std::map<std::string, std::string> config;
    config[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
    // Set log level
    switch (desc.debug_level) {
    case 0:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_NONE);
        break;
    case 1:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_ERROR);
        break;
    case 2:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
        break;
    case 3:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
        break;
    case 4:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
        break;
    case 5:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_TRACE);
        break;
    default:
        config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_NONE);
    };

    VPUXCompilerL0::VPUXCompilerL0* pvc = nullptr;
    try {
        pvc = new VPUXCompilerL0::VPUXCompilerL0(desc, config);
    } catch (const std::exception& error) {
        if (desc.debug_level > 0)
            std::cerr << error.what() << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        if (desc.debug_level > 0)
            std::cerr << "Internal exception!" << std::endl;
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    if (pvc != nullptr)
        *compiler = reinterpret_cast<vcl_compiler_handle_t>(pvc);
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerGetProperties(vcl_compiler_handle_t compiler, vcl_compiler_properties_t* properties) {
    if (!properties || !compiler) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXCompilerL0::VPUXCompilerL0* pvc = reinterpret_cast<VPUXCompilerL0::VPUXCompilerL0*>(compiler);
    *properties = pvc->getCompilerProp();
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                           vcl_executable_handle_t* executable) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    bool enableProfiling = false;

    const uint8_t* modelIR = desc.modelIRData;
    if (!compiler || !executable || !modelIR) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXCompilerL0::VPUXCompilerL0* pvc = reinterpret_cast<VPUXCompilerL0::VPUXCompilerL0*>(compiler);
    uint32_t debug_level = pvc->getCompilerDesc().debug_level;
    // Check exeDesc and create VPUXConfig
    std::map<std::string, std::string> config;
    // To avoid access violation
    std::string descOptions(desc.options, desc.optionsSize);
    if (debug_level > 0) {
        std::cout << "config: " << descOptions << std::endl;
    }
    std::size_t ips = descOptions.find(KEY_INPUTS_PRECISIONS);
    std::size_t ils = descOptions.find(KEY_INPUTS_LAYOUTS);
    std::size_t imls = descOptions.find(KEY_INPUTS_MODEL_LAYOUTS);
    std::size_t ops = descOptions.find(KEY_OUTPUTS_PRECISIONS);
    std::size_t ols = descOptions.find(KEY_OUTPUTS_LAYOUTS);
    std::size_t omls = descOptions.find(KEY_OUTPUTS_MODEL_LAYOUTS);
    std::size_t cs = descOptions.find(KEY_CONFIGS);
    std::vector<std::string> ioInfoOptions;
    if (ips != std::string::npos && ils != std::string::npos && ops != std::string::npos && ols != std::string::npos) {
        // Separate ioInfo to different section
        ioInfoOptions.push_back(descOptions.substr(ips, ils));
        if (imls != std::string::npos) {
            ioInfoOptions.push_back(descOptions.substr(ils, imls - ils));
            ioInfoOptions.push_back(descOptions.substr(imls, ops - imls));
        } else {
            ioInfoOptions.push_back(descOptions.substr(ils, ops - ils));
        }
        ioInfoOptions.push_back(descOptions.substr(ops, ols - ops));
        if (cs != std::string::npos) {
            if (omls != std::string::npos) {
                ioInfoOptions.push_back(descOptions.substr(ols, omls - ols));
                ioInfoOptions.push_back(descOptions.substr(omls, cs - omls));
            } else {
                ioInfoOptions.push_back(descOptions.substr(ols, cs - ols));
            }
        } else {
            if (omls != std::string::npos) {
                ioInfoOptions.push_back(descOptions.substr(ols, omls - ols));
                ioInfoOptions.push_back(descOptions.substr(omls));
            } else {
                ioInfoOptions.push_back(descOptions.substr(ols));
            }
        }
    } else {
        // Return error if the mandatory ioInfo options are not passed
        // Comment to skip ioInfo missing.
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    std::vector<std::string> options;
    if (cs != std::string::npos) {
        std::stringstream input(descOptions.substr(cs + strlen(KEY_CONFIGS)));
        std::string result;
        std::string temp = "";
        while (input >> result) {
            if (temp.compare("") == 0) {
                temp = result;
            } else {
                temp = temp + " " + result;
            }
            if (result[result.size() - 1] != '"') {
                continue;
            }
            options.push_back(temp);
            temp = "";
        }
        if (temp.compare("") != 0) {
            options.push_back(temp);
        }
    }
    if (debug_level > 0) {
        for (auto& op : options) {
            std::cout << "option : " << op << std::endl;
        }
    }
    // Set platform
    switch (pvc->getCompilerDesc().platform) {
    case VCL_PLATFORM_VPU3400:
        config[VPUX_CONFIG_KEY(PLATFORM)] = "3400";
        config[CONFIG_KEY(DEVICE_ID)] = "3400";
        break;
    case VCL_PLATFORM_VPU3700:
        config[VPUX_CONFIG_KEY(PLATFORM)] = "3700";
        config[CONFIG_KEY(DEVICE_ID)] = "3700";
        break;
    case VCL_PLATFORM_VPU3720:
        config[VPUX_CONFIG_KEY(PLATFORM)] = "3720";
        config[CONFIG_KEY(DEVICE_ID)] = "3720";
        break;
    default:
        config[VPUX_CONFIG_KEY(PLATFORM)] = "3700";
        config[CONFIG_KEY(DEVICE_ID)] = "3700";
    };

    // Options will overwrite default configs.
    try {
        for (auto& option : options) {
            if (option.find_first_not_of(' ') == std::string::npos)
                continue;
            size_t length = option.size();
            if (option[length - 1] == '\0')
                length--;
            std::size_t lastDelimPos = option.find_first_of('=');
            // Use 2 to skip =" , the format shall follow key="value"
            if (lastDelimPos == std::string::npos || lastDelimPos + 2 > length) {
                throw std::logic_error(option + " is in bad format!");
            }
            std::string key = option.substr(0, lastDelimPos);
            // For key="value", the val shall be value
            std::string val = option.substr(lastDelimPos + 2, length - 1 - (lastDelimPos + 2));
            if (debug_level > 3) {
                std::cout << "config options - key: " << key << " value: " << val << std::endl;
            }
            config[key] = val;
        }
    } catch (const std::exception& error) {
        if (debug_level > 0)
            std::cerr << error.what() << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        if (debug_level > 0)
            std::cerr << "Internal exception!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // Foce to use MLIR compiler.
    config[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);

    std::map<std::string, std::string>::iterator iter = config.find(CONFIG_KEY(LOG_LEVEL));
    if (iter != config.end()) {
        if (iter->second == CONFIG_VALUE(LOG_INFO))
            enableProfiling = true;
    }

    bool useOVAPI2 = false;
    std::map<std::string, std::string>::iterator iterAPI = config.find(KEY_VCL_OV_API_2);
    if (iterAPI != config.end()) {
        if (iterAPI->second == "YES") {
            useOVAPI2 = true;
        }
        config.erase(KEY_VCL_OV_API_2);
    }

    Config parsedConfig(pvc->getOptions());
    try {
        parsedConfig.update(config, OptionMode::CompileTime);
    } catch (const std::exception& error) {
        if (debug_level > 0)
            std::cerr << error.what() << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        if (debug_level > 0)
            std::cerr << "Internal exception!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    uint32_t offset = 0;
    vcl_version_info_t APIVersion;
    memcpy(&APIVersion, modelIR, sizeof(APIVersion));
    vcl_version_info_t currentAPIVersion = pvc->getCompilerProp().version;
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        if (debug_level > 0)
            std::cerr << "Unsupported IR API version! Val:" << APIVersion.major << "." << APIVersion.minor << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(vcl_version_info_t);
    uint32_t numOfElements = 0;
    memcpy(&numOfElements, modelIR + offset, sizeof(numOfElements));
    if (numOfElements >= maxNumberOfElements) {
        if (debug_level > 0)
            std::cerr << "Bad elements number in IR!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(numOfElements);
    uint64_t bufferSize = 0;
    memcpy(&bufferSize, modelIR + offset, sizeof(bufferSize));
    if (bufferSize == 0 || bufferSize >= maxSizeOfXML) {
        if (debug_level > 0)
            std::cerr << "Bad buffer size in IR!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(bufferSize);
    const uint8_t* buffer = modelIR + offset;
    offset += bufferSize;
    uint64_t weightsSize = 0;
    memcpy(&weightsSize, modelIR + offset, sizeof(weightsSize));
    if (weightsSize >= maxSizeOfWeights) {
        if (debug_level > 0)
            std::cerr << "Bad weights size in IR!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(weightsSize);
    const uint8_t* weights = modelIR + offset;
    if (offset + weightsSize > desc.modelIRSize) {
        if (debug_level > 0)
            std::cerr << "The IR content and size mismatch!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }

    VPUXCompilerL0::IOInfoV1 ioInfoV1;
    VPUXCompilerL0::IOInfoV2 ioInfoV2;
    try {
        if (!useOVAPI2) {
            ret = ioInfoV1.parse(ioInfoOptions, debug_level);
        } else {
            ret = ioInfoV2.parse(ioInfoOptions, debug_level);
        }
    } catch (const std::exception& error) {
        if (debug_level > 0)
            std::cerr << error.what() << std::endl;
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        if (debug_level > 0)
            std::cerr << "Internal exception! Can't parse ioInfo." << std::endl;
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (ret != VCL_RESULT_SUCCESS)
        return ret;

    // Create blob and set blob size.
    std::pair<VPUXCompilerL0::VPUXExecutableL0*, vcl_result_t> status;
    if (!useOVAPI2) {
        status =
                pvc->importNetworkV1(buffer, bufferSize, weights, weightsSize, parsedConfig, ioInfoV1, enableProfiling);
    } else {
        status =
                pvc->importNetworkV2(buffer, bufferSize, weights, weightsSize, parsedConfig, ioInfoV2, enableProfiling);
    }
    if (status.second != VCL_RESULT_SUCCESS) {
        *executable = NULL;
        return status.second;
    } else {
        VPUXCompilerL0::VPUXExecutableL0* pve = status.first;
        ret = pve->serializeNetwork();
        if (ret != VCL_RESULT_SUCCESS) {
            delete pve;
            *executable = NULL;
            return ret;
        }
        *executable = reinterpret_cast<vcl_executable_handle_t>(pve);
    }
    return ret;
}

DLLEXPORT vcl_result_t vclExecutableGetSerializableBlob(vcl_executable_handle_t executable, uint8_t* blobBuffer,
                                                        uint64_t* blobSize) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    if (!blobSize || !executable) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXCompilerL0::VPUXExecutableL0* pve = reinterpret_cast<VPUXCompilerL0::VPUXExecutableL0*>(executable);
    if (!blobBuffer) {
        ret = pve->getNetworkSize(blobSize);
    } else {
        ret = pve->exportNetwork(blobBuffer, *blobSize);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableDestroy(vcl_executable_handle_t executable) {
    if (executable) {
        VPUXCompilerL0::VPUXExecutableL0* pve = reinterpret_cast<VPUXCompilerL0::VPUXExecutableL0*>(executable);
        delete pve;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerDestroy(vcl_compiler_handle_t compiler) {
    if (compiler) {
        VPUXCompilerL0::VPUXCompilerL0* pvc = reinterpret_cast<VPUXCompilerL0::VPUXCompilerL0*>(compiler);
        delete pvc;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingCreate(p_vcl_profiling_input_t profilingInput,
                                                      vcl_profiling_handle_t* profilingHandle) {
    if (!profilingInput || !profilingHandle) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXCompilerL0::VPUXProfilingL0* profHandle = new (std::nothrow) VPUXCompilerL0::VPUXProfilingL0(profilingInput);
    if (!profHandle) {
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }

    vcl_result_t result = profHandle->preprocess();
    if (result != VCL_RESULT_SUCCESS) {
        return result;
    }

    *profilingHandle = reinterpret_cast<vcl_profiling_handle_t>(profHandle);
    return VCL_RESULT_SUCCESS;
}
DLLEXPORT vcl_result_t VCL_APICALL vclGetDecodedProfilingBuffer(vcl_profiling_handle_t profilingHandle,
                                                                vcl_profiling_request_type_t requestType,
                                                                p_vcl_profiling_output_t profilingOutput) {
    if (!profilingHandle || !profilingOutput) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXCompilerL0::VPUXProfilingL0* prof = reinterpret_cast<VPUXCompilerL0::VPUXProfilingL0*>(profilingHandle);
    switch (requestType) {
    case VCL_PROFILING_LAYER_LEVEL:
        return prof->getLayerInfo(profilingOutput);
    case VCL_PROFILING_TASK_LEVEL:
        return prof->getTaskInfo(profilingOutput);
    case VCL_PROFILING_RAW:
        return prof->getRawInfo(profilingOutput);
    default:
        std::cerr << "Request type is not supported." << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingDestroy(vcl_profiling_handle_t profilingHandle) {
    if (profilingHandle) {
        VPUXCompilerL0::VPUXProfilingL0* pvp = reinterpret_cast<VPUXCompilerL0::VPUXProfilingL0*>(profilingHandle);
        delete pvp;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingGetProperties(vcl_profiling_handle_t profilingHandle,
                                                             vcl_profiling_properties_t* properties) {
    if (!profilingHandle || !properties) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXCompilerL0::VPUXProfilingL0* pvp = reinterpret_cast<VPUXCompilerL0::VPUXProfilingL0*>(profilingHandle);
    *properties = pvp->getProperties();
    return VCL_RESULT_SUCCESS;
}

#ifdef __cplusplus
}
#endif

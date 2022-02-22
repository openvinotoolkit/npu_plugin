//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "VPUXCompilerL0.h"

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
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

#define COMPILER_MAJOR 1
#define COMPILER_MINOR 3
static const char* COMPILER_VERSION = xstr(COMPILER_MAJOR) "." xstr(COMPILER_MINOR);

#define PROFILING_MAJOR 1
#define PROFILING_MINOR 0

const uint32_t maxNumberOfElements = 10;
const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

constexpr uint16_t  = 1;
constexpr uint16_t PROFILING_MINOR = 0;

using namespace vpux;

namespace VPUXCompilerL0 {

struct IOInfo {
    InferenceEngine::Precision getPrecision(const vcl_tensor_precision_t value) {
        // Remove some IE precisions to follow checkNetworkPrecision().
        // Removed U64, I64, BF16, U16, I16, BOOL.
        static const std::unordered_map<vcl_tensor_precision_t, InferenceEngine::Precision> supported_precisions = {
                {VCL_TENSOR_PRECISION_FP32, InferenceEngine::Precision::FP32},
                {VCL_TENSOR_PRECISION_FP16, InferenceEngine::Precision::FP16},
                {VCL_TENSOR_PRECISION_UINT32, InferenceEngine::Precision::U32},
                {VCL_TENSOR_PRECISION_INT32, InferenceEngine::Precision::I32},
                {VCL_TENSOR_PRECISION_UINT8, InferenceEngine::Precision::U8},
                {VCL_TENSOR_PRECISION_INT8, InferenceEngine::Precision::I8},
        };

        const auto precision = supported_precisions.find(value);
        if (precision == supported_precisions.end()) {
            // For unknown precision, use default value.
            return InferenceEngine::Precision::UNSPECIFIED;
        } else {
            return precision->second;
        }
    }

    InferenceEngine::Layout getLayout(const vcl_tensor_layout_t value) {
        static const std::unordered_map<vcl_tensor_layout_t, InferenceEngine::Layout> supported_layouts = {
                {VCL_TENSOR_LAYOUT_NCDHW, InferenceEngine::Layout::NCDHW},
                {VCL_TENSOR_LAYOUT_NDHWC, InferenceEngine::Layout::NDHWC},
                {VCL_TENSOR_LAYOUT_NCHW, InferenceEngine::Layout::NCHW},
                {VCL_TENSOR_LAYOUT_NHWC, InferenceEngine::Layout::NHWC},
                {VCL_TENSOR_LAYOUT_CHW, InferenceEngine::Layout::CHW},
                {VCL_TENSOR_LAYOUT_HWC, InferenceEngine::Layout::HWC},
                {VCL_TENSOR_LAYOUT_NC, InferenceEngine::Layout::NC},
                {VCL_TENSOR_LAYOUT_C, InferenceEngine::Layout::C},
        };
        const auto layout = supported_layouts.find(value);
        if (layout == supported_layouts.end()) {
            // For unknown layout, use default value.
            return InferenceEngine::Layout::ANY;
        } else {
            return layout->second;
        }
    }

    InferenceEngine::Precision inPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::Precision outPrc;
    InferenceEngine::Layout outLayout;
};

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

    std::pair<VPUXExecutableL0*, vcl_result_t> importNetwork(const uint8_t* buffer, uint64_t bufferSize,
                                                             const uint8_t* weights, uint64_t weightsSize,
                                                             Config& vpuxConfig, const IOInfo& ioInfo,
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

    Config parsedConfig(_options);
    parsedConfig.update(config, OptionMode::CompileTime);
    _compiler = Compiler::create(parsedConfig);

    _compilerDesc = desc;
    _compilerProp.id = COMPILER_VERSION;
    _compilerProp.version.major = COMPILER_MAJOR;
    _compilerProp.version.minor = COMPILER_MINOR;
    _compilerProp.supportedOpsets = 7;
}

std::pair<VPUXExecutableL0*, vcl_result_t> VPUXCompilerL0::importNetwork(const uint8_t* buffer, uint64_t bufferSize,
                                                                         const uint8_t* weights, uint64_t weightsSize,
                                                                         Config& config, const IOInfo& ioInfo,
                                                                         bool enableProfiling) {
    if (buffer == nullptr || weights == nullptr) {
        return std::pair<VPUXExecutableL0*, vcl_result_t>(nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT);
    }
    std::string model(buffer, buffer + bufferSize);
    InferenceEngine::MemoryBlob::Ptr weightsBlob;
    if (weightsSize != 0) {
        InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {weightsSize},
                                               InferenceEngine::Layout::C);
        weightsBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
        weightsBlob->allocate();
        memcpy(weightsBlob->rwmap().as<uint8_t*>(), weights, weightsSize);
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
        for (const auto& in : cnnNet.getInputsInfo()) {
            if (ioInfo.inLayout != InferenceEngine::Layout::ANY &&
                // cannot setLayout for fully-dynamic network
                !in.second->getPartialShape().rank().is_dynamic()) {
                in.second->setLayout(ioInfo.inLayout);
            }
            if (ioInfo.inPrc != InferenceEngine::Precision::UNSPECIFIED) {
                in.second->setPrecision(ioInfo.inPrc);
            }
        }

        for (const auto& out : cnnNet.getOutputsInfo()) {
            if (ioInfo.outLayout != InferenceEngine::Layout::ANY &&
                // cannot setLayout for fully-dynamic network
                !out.second->getPartialShape().rank().is_dynamic()) {
                out.second->setLayout(ioInfo.outLayout);
            }
            if (ioInfo.outPrc != InferenceEngine::Precision::UNSPECIFIED) {
                out.second->setPrecision(ioInfo.outPrc);
            }
        }

        networkDesc = _compiler->compile(cnnNet.getFunction(), cnnNet.getName(), cnnNet.getInputsInfo(),
                                         cnnNet.getOutputsInfo(), config);
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
              _profSize(profInput->profSize)
    { }
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
        _taskInfo = profiling::getTaskInfo(_blobData, _blobSize, _profData, _profSize, profiling::TaskType::ALL);
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
    // Check desc here
    if (desc.platform != VCL_PLATFORM_VPU3400 && desc.platform != VCL_PLATFORM_VPU3700 || compiler == nullptr) {
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
        std::cerr << error.what() << std::endl;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
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

    // Check exeDesc and create VPUXConfig
    std::map<std::string, std::string> config;
    // To avoid access violation
    std::string descOptions(desc.options, desc.optionsSize);
    std::stringstream input(descOptions);
    std::string result;
    std::vector<std::string> options;
    while (input >> result) {
        options.push_back(result);
    }

    // Set platform
    VPUXCompilerL0::VPUXCompilerL0* pvc = reinterpret_cast<VPUXCompilerL0::VPUXCompilerL0*>(compiler);
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
    size_t count = options.size();
    for (size_t i = 0; i + 1 < count;) {
        config[options[i]] = options[i + 1];
        i += 2;
    }
    // Foce to use MLIR compiler.
    config[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);

    std::map<std::string, std::string>::iterator iter = config.find(CONFIG_KEY(LOG_LEVEL));
    if (iter != config.end()) {
        if (iter->second == CONFIG_VALUE(LOG_INFO))
            enableProfiling = true;
    }

    Config parsedConfig(pvc->getOptions());
    parsedConfig.update(config, OptionMode::CompileTime);

    uint32_t offset = 0;
    vcl_version_info_t APIVersion;
    memcpy(&APIVersion, modelIR, sizeof(APIVersion));
    vcl_version_info_t currentAPIVersion = pvc->getCompilerProp().version;
    if (APIVersion.major != currentAPIVersion.major || APIVersion.minor != currentAPIVersion.minor) {
        std::cerr << "Unsupported IR API version!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(vcl_version_info_t);
    uint32_t numOfElements = 0;
    memcpy(&numOfElements, modelIR + offset, sizeof(numOfElements));
    if (numOfElements >= maxNumberOfElements) {
        std::cerr << "Bad elements number in IR!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(numOfElements);
    uint64_t bufferSize = 0;
    memcpy(&bufferSize, modelIR + offset, sizeof(bufferSize));
    if (bufferSize == 0 || bufferSize >= maxSizeOfXML) {
        std::cerr << "Bad buffer size in IR!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(bufferSize);
    const uint8_t* buffer = modelIR + offset;
    offset += bufferSize;
    uint64_t weightsSize = 0;
    memcpy(&weightsSize, modelIR + offset, sizeof(weightsSize));
    if (weightsSize >= maxSizeOfWeights) {
        std::cerr << "Bad weights size in IR!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }
    offset += sizeof(weightsSize);
    const uint8_t* weights = modelIR + offset;
    if (offset + weightsSize > desc.modelIRSize) {
        std::cerr << "The IR content and size mismatch!" << std::endl;
        return VCL_RESULT_ERROR_INVALID_IR;
    }

    VPUXCompilerL0::IOInfo ioInfo;
    ioInfo.inPrc = ioInfo.getPrecision(desc.inPrc);
    ioInfo.inLayout = ioInfo.getLayout(desc.inLayout);
    ioInfo.outPrc = ioInfo.getPrecision(desc.outPrc);
    ioInfo.outLayout = ioInfo.getLayout(desc.outLayout);

    // Create blob and set blob size.
    auto status = pvc->importNetwork(buffer, bufferSize, weights, weightsSize, parsedConfig, ioInfo, enableProfiling);
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

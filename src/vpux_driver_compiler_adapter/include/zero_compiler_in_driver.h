//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "iexternal_compiler.h"
#include "network_description.h"

#include <ze_api.h>
#include <ze_graph_ext.h>
#include <type_traits>

namespace vpux {
namespace driverCompilerAdapter {

#define NotSupportLogHandle(T)                                                                                 \
    (std::is_same<T, ze_graph_dditable_ext_t>::value || std::is_same<T, ze_graph_dditable_ext_1_1_t>::value || \
     std::is_same<T, ze_graph_dditable_ext_1_2_t>::value || std::is_same<T, ze_graph_dditable_ext_1_3_t>::value)

using NetworkInputs = DataMap;
using NetworkOutputs = DataMap;
using DeviceInputs = DataMap;
using DeviceOutputs = DataMap;

/**
 * Adapter to use CiD through ZeroAPI
 */
template <typename TableExtension>
class LevelZeroCompilerInDriver final : public IExternalCompiler {
public:
    LevelZeroCompilerInDriver(const char* extName, ze_driver_handle_t driverHandle);
    ~LevelZeroCompilerInDriver() override;

    size_t getSupportedOpset() override;

    std::shared_ptr<INetworkDescription> compileIR(const std::string& graphName, const std::vector<char>& xml,
                                                   const std::vector<char>& weights,
                                                   const InferenceEngine::InputsDataMap& inputsInfo,
                                                   const InferenceEngine::OutputsDataMap& outputsInfo,
                                                   const vpux::Config& config) final;

    std::shared_ptr<INetworkDescription> parseBlob(const std::string& graphName, const std::vector<char>& blob,
                                                   const vpux::Config& config) final;
    static std::string serializeIOInfo(const InferenceEngine::InputsDataMap& inputsInfo,
                                       const InferenceEngine::OutputsDataMap& outputsInfo);

private:
    NetworkMeta getNetworkMeta(ze_graph_handle_t graphHandle);

    std::vector<uint8_t> serializeIR(const std::vector<char>& xml, const std::vector<char>& weights);
    template <typename T>
    void getNetDevInOutputs(DataMap& netInputs, DataMap& devInputs, DataMap& netOutputs, DataMap& devOutputs,
                            const T& arg);

    void getMetaData(TableExtension* graphDdiTableExt, ze_graph_handle_t graphHandle, uint32_t index,
                     DataMap& netInputs, DataMap& devInputs, DataMap& netOutputs, DataMap& devOutputs,
                     std::vector<OVRawNode>& ovResults, std::vector<OVRawNode>& ovParameters);

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportLogHandle(T), bool> = true>
    std::string getLatestBuildError();

    template <typename T = TableExtension, typename std::enable_if_t<NotSupportLogHandle(T), bool> = true>
    std::string getLatestBuildError() {
        return "";
    }

private:
    ze_driver_handle_t _driverHandle = nullptr;
    ze_device_handle_t _deviceHandle = nullptr;
    ze_context_handle_t _context = nullptr;

    TableExtension* _graphDdiTableExt = nullptr;
    Logger _logger;
};

template <typename TableExtension>
LevelZeroCompilerInDriver<TableExtension>::LevelZeroCompilerInDriver(const char* extName,
                                                                     ze_driver_handle_t driverHandle)
        : _driverHandle(driverHandle), _logger("LevelZeroCompilerInDriver", LogLevel::Warning) {
    // Load our graph extension
    auto result =
            zeDriverGetExtensionFunctionAddress(_driverHandle, extName, reinterpret_cast<void**>(&_graphDdiTableExt));

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to initialize zeDriver. Error code: " << std::hex << result;
    }

    uint32_t deviceCount = 1;
    // Get our target device
    result = zeDeviceGet(_driverHandle, &deviceCount, &_deviceHandle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get device. Error code: " << std::hex << result;
    }

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(_driverHandle, &contextDesc, &_context);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to initialize context for device. Error code: " << std::hex << result;
    }
}

}  // namespace driverCompilerAdapter
}  // namespace vpux

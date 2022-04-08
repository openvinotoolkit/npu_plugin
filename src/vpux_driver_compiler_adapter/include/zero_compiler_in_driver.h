//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "iexternal_compiler.h"
#include "network_description.h"
#include "ze_api.h"
#include "ze_graph_ext.h"

namespace vpux {
namespace driverCompilerAdapter {

using NetworkInputs = DataMap;
using NetworkOutputs = DataMap;
using DeviceInputs = DataMap;
using DeviceOutputs = DataMap;

/**
 * Adapter to use CiD through ZeroAPI
 */
class LevelZeroCompilerInDriver final : public IExternalCompiler {
public:
    LevelZeroCompilerInDriver();
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
    NetworkMeta getNetworkMeta(ze_graph_handle_t graph_handle);

    std::vector<uint8_t> serializeIR(const std::vector<char>& xml, const std::vector<char>& weights);

private:
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;

    Logger _logger;
};

}  // namespace driverCompilerAdapter
}  // namespace vpux

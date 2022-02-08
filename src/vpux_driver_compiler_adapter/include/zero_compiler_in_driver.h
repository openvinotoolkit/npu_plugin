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

#include "iexternal_compiler.h"
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

    std::shared_ptr<INetworkDescription> compileIR(const std::string& graphName, const std::vector<char>& xml, const std::vector<char>& weights,
                const InferenceEngine::InputsDataMap& inputsInfo, const InferenceEngine::OutputsDataMap& outputsInfo,
                const vpux::Config& config) final;

    std::shared_ptr<INetworkDescription> parseBlob(const std::string& graphName, const std::vector<char>& blob, const vpux::Config& config) final;

private:
    std::tuple<const NetworkInputs, const NetworkOutputs, const DeviceInputs, const DeviceOutputs> getNetworkMeta(
            ze_graph_handle_t graph_handle);

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

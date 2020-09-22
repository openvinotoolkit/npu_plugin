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

#include <ie_icnn_network.hpp>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__arm__) || defined(__aarch64__)
#include <NnCorePlg.h>
#include <NnXlinkPlg.h>
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

#include <kmb_config.h>

#include <vpux.hpp>

namespace vpu {
namespace KmbPlugin {

class KmbNNCoreExecutor : public vpux::Executor {
public:
    using Ptr = std::shared_ptr<KmbNNCoreExecutor>;

    virtual ~KmbNNCoreExecutor();
    KmbNNCoreExecutor(const vpux::NetworkDescription::Ptr& networkDescription,
        const std::shared_ptr<vpux::Allocator>& allocator, const KmbConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void push(const InferenceEngine::BlobMap& inputs, const vpux::PreprocMap& preProcMap) override;

    void pull(InferenceEngine::BlobMap& outputs) override;
    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

private:
    vpux::NetworkDescription::Ptr _networkDescription;
    std::shared_ptr<vpux::Allocator> _allocator;
    const KmbConfig& _config;
    Logger::Ptr _logger;

#if defined(__arm__) || defined(__aarch64__)
    std::shared_ptr<NnCorePlg> _nnCorePlg;
    std::shared_ptr<NnXlinkPlg> _nnXlinkPlg;

    void* blob_file = nullptr;
    std::shared_ptr<BlobHandle_t> _blobHandle;

    std::shared_ptr<Pipeline> _pipe;
#endif
    void initVpualObjects();
    void allocateGraph(const std::vector<char>& compiledNetwork);
    void deallocateGraph();

    InferenceEngine::Blob::Ptr prepareInputForInference(
        const InferenceEngine::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& deviceDesc);
    uint32_t extractPhysAddrForInference(const InferenceEngine::BlobMap& inputs);

    InferenceEngine::BlobMap extractOutputsFromPhysAddr(uint32_t physAddr);
    void repackDeviceOutputsToNetworkOutputs(
        const InferenceEngine ::BlobMap& deviceOutputs, InferenceEngine::BlobMap& networkOutputs);

    std::vector<void*> _scratchBuffers;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _inputBuffer;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _outputBuffer;

    std::vector<uint32_t> _outputPhysAddrs;
};

}  // namespace KmbPlugin
}  // namespace vpu

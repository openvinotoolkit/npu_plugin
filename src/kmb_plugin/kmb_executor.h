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
#include <GraphManagerPlg.h>
#include <MemAllocator.h>
#include <NNFlicPlg.h>
#include <PlgInferenceInput.h>
#include <PlgInferenceOutput.h>
#include <PlgStreamResult.h>
#include <PlgTensorSource.h>
#include <Pool.h>
#include <cma_allocation_helper.h>
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

#include <kmb_config.h>

#include <vpux.hpp>

#include "kmb_allocator.h"

namespace vpu {
namespace KmbPlugin {

class KmbExecutor : public vpux::Executor {
public:
    using Ptr = std::shared_ptr<KmbExecutor>;

    virtual ~KmbExecutor();
    KmbExecutor(const vpux::NetworkDescription::Ptr& networkDescription, const KmbAllocator::Ptr& allocator,
        const KmbConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void pull(InferenceEngine::BlobMap& outputs) override;
    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) override;

private:
    vpux::NetworkDescription::Ptr _networkDescription;
    KmbAllocator::Ptr _allocator;
    const KmbConfig& _config;
    Logger::Ptr _logger;

#if defined(__arm__) || defined(__aarch64__)
    std::shared_ptr<GraphManagerPlg> gg;
    std::shared_ptr<PlgTensorSource> plgTensorInput_;
    std::shared_ptr<PlgStreamResult> plgTensorOutput_;
    std::shared_ptr<PlgInferenceInput> plgInferenceInput_;
    std::shared_ptr<PlgInferenceOutput> plgInferenceOutput_;
    std::shared_ptr<RgnAllocator> RgnAlloc;
    std::shared_ptr<HeapAllocator> HeapAlloc;

    std::shared_ptr<NNFlicPlg> nnPl;

    void* blob_file = nullptr;
    std::shared_ptr<BlobHandle_t> BHandle;

    std::shared_ptr<PlgPool<TensorMsg>> plgPoolOutputs;
    std::shared_ptr<PlgPool<InferenceMsg>> plgPoolInferenceMsg;

    std::shared_ptr<Pipeline> pipe;
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

    // _inferenceId is used to satisfy VPUAL API which requires to pass some id for each inference
    // there are no contraints on a value passed, so we pass id=1 each inference
    std::unique_ptr<uint32_t, std::function<void(uint32_t*)>> _inferenceId;
};

}  // namespace KmbPlugin
}  // namespace vpu

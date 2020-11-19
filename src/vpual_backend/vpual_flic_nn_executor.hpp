//
// Copyright 2019-2020 Intel Corporation.
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

// FIXME: get back config
// #include <kmb_config.h>
#include <vpu/utils/logger.hpp>
#include <vpux.hpp>
#include <vpux_config.hpp>

#include "vpual_config.hpp"
#include "vpusmm_allocator.hpp"

namespace ie = InferenceEngine;

namespace vpux {

class VpualFlicNNExecutor final : public vpux::Executor {
public:
    using Ptr = std::shared_ptr<VpualFlicNNExecutor>;

    virtual ~VpualFlicNNExecutor();
    VpualFlicNNExecutor(const vpux::NetworkDescription::Ptr& networkDescription, const VpusmmAllocator::Ptr& allocator,
        const uint32_t deviceId, const VpualConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    void pull(InferenceEngine::BlobMap& outputs) override;
    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const PreprocMap& preProcMap) const override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

private:
    vpux::NetworkDescription::Ptr _networkDescription;
    VpusmmAllocator::Ptr _allocator;
    const VpualConfig& _config;
    vpu::Logger::Ptr _logger;

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
    void initVpualObjects(const uint32_t deviceId);
    void allocateGraph(const std::vector<char>& compiledNetwork);
    void deallocateGraph();

    ie::Blob::Ptr prepareInputForInference(
        const ie::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& deviceDesc);
    uint32_t extractPhysAddrForInference(const ie::BlobMap& inputs);

    ie::BlobMap extractOutputsFromPhysAddr(uint32_t physAddr);
    void repackDeviceOutputsToNetworkOutputs(
        const InferenceEngine ::BlobMap& deviceOutputs, InferenceEngine::BlobMap& networkOutputs);

    std::vector<void*> _scratchBuffers;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _inputBuffer;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _outputBuffer;
    std::unique_ptr<uint32_t, std::function<void(uint32_t*)>> _inferenceId;
};

}  // namespace vpux

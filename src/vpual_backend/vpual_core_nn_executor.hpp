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
#include <NnCorePlg.h>
#include <NnXlinkPlg.h>
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

class VpualCoreNNExecutor final : public vpux::Executor {
public:
    using Ptr = std::shared_ptr<VpualCoreNNExecutor>;

    virtual ~VpualCoreNNExecutor();
    VpualCoreNNExecutor(const vpux::NetworkDescription::Ptr& networkDescription, const VpusmmAllocator::Ptr& allocator,
        const VpusmmAllocator::Ptr& csramAllocator, const uint32_t deviceId, const VpualConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    void pull(InferenceEngine::BlobMap& outputs) override;
    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

private:
    vpux::NetworkDescription::Ptr _networkDescription;
    VpusmmAllocator::Ptr _allocator;
    VpusmmAllocator::Ptr _csramAllocator;
    const VpualConfig& _config;
    vpu::Logger::Ptr _logger;

#if defined(__arm__) || defined(__aarch64__)
    std::unique_ptr<NnXlinkPlg> _nnXlinkPlg = nullptr;
    std::unique_ptr<NnCorePlg, std::function<void(NnCorePlg*)>> _nnCorePlg = nullptr;
    // pipeline has to be deleted before NNCore plug-in
    // otherwise it leads to 'Bus error'
    std::unique_ptr<Pipeline, std::function<void(Pipeline*)>> _pipe = nullptr;
    std::unique_ptr<void, std::function<void(void*)>> blob_file = nullptr;
    std::unique_ptr<BlobHandle_t> _blobHandle = nullptr;

#endif
    void allocateGraph(const std::vector<char>& compiledNetwork);

    ie::Blob::Ptr prepareInputForInference(
        const ie::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& deviceDesc);
    uint32_t extractPhysAddrForInference(const ie::BlobMap& inputs);

    ie::BlobMap extractOutputsFromPhysAddr(uint32_t physAddr);
    void repackDeviceOutputsToNetworkOutputs(
        const InferenceEngine ::BlobMap& deviceOutputs, InferenceEngine::BlobMap& networkOutputs);

    std::vector<void*> _scratchBuffers;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _preFetchBuffer;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _inputBuffer;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _outputBuffer;

    std::vector<uint32_t> _outputPhysAddrs;
};

}  // namespace vpux

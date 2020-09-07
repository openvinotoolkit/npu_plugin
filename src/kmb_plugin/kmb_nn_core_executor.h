//
// Copyright 2020 Intel Corporation.
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

#include "kmb_allocator.h"
#include "kmb_executor.h"

namespace vpu {
namespace KmbPlugin {

class KmbNNCoreExecutor : public KmbExecutor {
public:
    using Ptr = std::shared_ptr<KmbNNCoreExecutor>;

    explicit KmbNNCoreExecutor(const KmbConfig& config);
    virtual ~KmbNNCoreExecutor() = default;

    virtual void allocateGraph(const std::vector<char>& graphFileContent, const ie::InputsDataMap& networkInputs,
        const ie::OutputsDataMap& networkOutputs, bool newFormat);

    virtual void deallocateGraph();

    virtual void queueInference(void* input_data, size_t input_bytes);

    virtual void getResult(void* result_data, unsigned int result_bytes);

    const KmbConfig& _config;

private:
    Logger::Ptr _logger;
#if defined(__arm__) || defined(__aarch64__)
    std::shared_ptr<NnCorePlg> _nnCorePlg;
    std::shared_ptr<NnXlinkPlg> _nnXlinkPlg;

    void* blob_file = nullptr;
    std::shared_ptr<BlobHandle_t> _blobHandle;

    std::shared_ptr<Pipeline> _pipe;
#endif
    std::shared_ptr<KmbAllocator> allocator;
    void initVpualObjects();

    const int xlinkChannel = 0;

    uint32_t* _inferenceVirtAddr;
    std::vector<void*> _scratchBuffers;
    std::vector<uint32_t> _inputSizes;
    std::vector<uint32_t> _outputPhysAddrs;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _outputBuffer;
};

}  // namespace KmbPlugin
}  // namespace vpu

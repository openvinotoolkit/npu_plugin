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

#include <kmb_config.h>

#include <ie_icnn_network.hpp>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kmb_allocator.h"

namespace vpu {
namespace KmbPlugin {

class KmbExecutor {
public:
    using Ptr = std::shared_ptr<KmbExecutor>;

    virtual ~KmbExecutor() = default;

    virtual void allocateGraph(const std::vector<char>& graphFileContent, const ie::InputsDataMap& networkInputs,
        const ie::OutputsDataMap& networkOutputs, bool newFormat) = 0;

    virtual void deallocateGraph() = 0;

    virtual void queueInference(void* input_data, size_t input_bytes) = 0;

    virtual void getResult(void* result_data, unsigned int result_bytes) = 0;

    virtual const InferenceEngine::InputsDataMap& getRuntimeInputs() const { return _runtimeInputs; }
    virtual const InferenceEngine::OutputsDataMap& getRuntimeOutputs() const { return _runtimeOutputs; }

protected:
    InferenceEngine::InputsDataMap _runtimeInputs;
    InferenceEngine::OutputsDataMap _runtimeOutputs;
};

}  // namespace KmbPlugin
}  // namespace vpu

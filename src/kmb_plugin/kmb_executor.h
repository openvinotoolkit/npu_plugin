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

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iomanip>
#include <utility>

#ifdef ENABLE_VPUAL
#include <GraphManagerPlg.h>
#include <PlgStreamResult.h>
#include <PlgTensorSource.h>

#include <NNFlicPlg.h>
#include <Pool.h>
#include <cma_allocation_helper.h>
#include <MemAllocator.h>
#include <mvMacros.h>
#endif

#include <kmb_config.h>
#include "kmb_blob_parser.hpp"

namespace vpu {
namespace KmbPlugin {

class KmbExecutor {
    Logger::Ptr _log;
    unsigned int _numStages = 0;
#ifdef ENABLE_VPUAL
    std::shared_ptr<GraphManagerPlg> gg;
    std::shared_ptr<PlgTensorSource> plgTensorInput_;
    std::shared_ptr<PlgStreamResult> plgTensorOutput_;
    std::shared_ptr<HeapAllocator> HeapAlloc;

    std::shared_ptr<NNFlicPlg> nnPl;

    std::shared_ptr<CmaData> blob_file;
    std::shared_ptr<CmaData> input_tensor;
    std::shared_ptr<CmaData> output_tensor;
    std::shared_ptr<BlobHandle_t> BHandle;

    std::shared_ptr<PlgPool<TensorMsg>> plgPoolA;
    std::shared_ptr<PlgPool<TensorMsg>> plgPoolB;

    std::shared_ptr<Pipeline> pipe;

    const std::shared_ptr<KmbConfig>& _config;
#endif

public:
    KmbExecutor(const Logger::Ptr& log, const std::shared_ptr<KmbConfig>& config);
    ~KmbExecutor() = default;

    void allocateGraph(const std::vector<char> &graphFileContent, const char* networkName);

    void deallocateGraph();

    void queueInference(void *input_data, size_t input_bytes, void *result_data, size_t result_bytes);

    void getResult(void *result_data, unsigned int result_bytes);

    const std::shared_ptr<KmbConfig>& _config;
};

typedef std::shared_ptr<KmbExecutor> KmbExecutorPtr;

}  // namespace KmbPlugin
}  // namespace vpu

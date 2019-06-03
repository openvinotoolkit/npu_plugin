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

#include <mvnc.h>
#include "GraphManagerPlg.h"
#include "PlgStreamResult.h"
#include "PlgTensorSource.h"
#include <kmb_config.h>
#include "kmb_blob_parser.hpp"

namespace vpu {
namespace KmbPlugin {

struct GraphDesc {
    ncGraphHandle_t *_graphHandle = nullptr;

    ncTensorDescriptor_t _inputDesc = {};
    ncTensorDescriptor_t _outputDesc = {};

    ncFifoHandle_t *_inputFifoHandle = nullptr;
    ncFifoHandle_t *_outputFifoHandle = nullptr;
};

struct DeviceDesc {
    int _executors = 0;
    int _maxExecutors = 0;
    ncDevicePlatform_t _platform = ANY_PLATFORM;
    int _deviceIdx = -1;
    ncDeviceHandle_t *_deviceHandle = nullptr;

    bool isBooted() const {
        return _deviceHandle != nullptr;
    }
    bool isEmpty() const {
        return _executors == 0;
    }
    bool isAvailable() const {
        return _executors < _maxExecutors;
    }
};

typedef std::shared_ptr<DeviceDesc> DevicePtr;


class KmbExecutor {
    Logger::Ptr _log;
    unsigned int _numStages = 0;

public:
    KmbExecutor(bool forceReset, const LogLevel& vpuLogLevel, const Logger::Ptr& log) : _log(log) {}
    ~KmbExecutor() = default;

    void allocateGraph(DevicePtr &device,
                       GraphDesc &graphDesc,
                       const std::vector<char> &graphFileContent,
                       const MVCNN::SummaryHeader *graphHeaderDesc,
                       size_t numStages,
                       const char* networkName);

    void deallocateGraph(DevicePtr &device, GraphDesc &graphDesc);

    void queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                        void *result_data, size_t result_bytes);

    void getResult(GraphDesc &graphDesc, void *result_data, unsigned int result_bytes);

    std::shared_ptr<GraphManagerPlg> gg;
    std::shared_ptr<PlgTensorSource> plgTensorInput_;
    std::shared_ptr<PlgStreamResult> plgTensorOutput_;

    std::shared_ptr<HeapAllocator> HeapAlloc;

private:
    int flic_pipeline(int graphId, BlobHandle_t* BHandle, int nThreads, int nShaves);
};

typedef std::shared_ptr<KmbExecutor> KmbExecutorPtr;

}  // namespace KmbPlugin
}  // namespace vpu

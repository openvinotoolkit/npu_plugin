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

#include <cstdlib>

#include <ie_icnn_network.hpp>
#include <vpu/graph_transformer.hpp>

#include "graphfile_generated.h"

namespace vpu {
namespace KmbPlugin {

class KmbBlob {
public:
//  FlatBuffer blob parser to get information about network inputs/outputs
//  from flatbuffer blob file loaded by ImportNetwork method

    KmbBlob(const void *data, size_t size);

    const InferenceEngine::InputsDataMap& getNetworkInputs() const { return _networkInputs; }
    const InferenceEngine::OutputsDataMap& getNetworkOutputs() const { return _networkOutputs; }

    const DataInfo& getInputInfo()  const { return _inputInfo; }
    const DataInfo& getOutputInfo() const { return _outputInfo; }

    uint32_t getStageCount() const { return _blobHeader->layer_count(); }
    uint32_t getVersionMinor() const { return _blobHeader->version()->minorV(); }
    uint32_t getVersionMajor() const { return _blobHeader->version()->majorV(); }

    const MVCNN::SummaryHeader* getHeader() const { return _blobHeader; }

private:
    InferenceEngine::InputsDataMap  _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;

    DataInfo _inputInfo;
    DataInfo _outputInfo;

    uint32_t _stageCount;
    const MVCNN::SummaryHeader* _blobHeader;
};

}  // namespace KmbPlugin
}  // namespace vpu

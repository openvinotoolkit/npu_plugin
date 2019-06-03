//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <vector>
#include <utility>

#include <ie_input_info.hpp>
#include <ie_icnn_network.hpp>

#include <vpu/backend/blob_format.hpp>
#include <vpu/model/data_desc.hpp>
#include <vpu/graph_transformer.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class BlobReader {
public:
    BlobReader() = default;

    void parse(const std::vector<char>& blob);

    const ie::InputsDataMap& getNetworkInputs() const { return _networkInputs; }
    const ie::OutputsDataMap& getNetworkOutputs() const { return _networkOutputs; }

    uint32_t getStageCount() const { return _blobHeader.stages_count; }

    uint32_t getMagicNumber() const { return _blobHeader.magic_number; }

    uint32_t getVersionMajor() const { return _blobHeader.blob_ver_major; }
    uint32_t getVersionMinor() const { return _blobHeader.blob_ver_minor; }

    const DataInfo& getInputInfo()  const { return _inputInfo; }
    const DataInfo& getOutputInfo() const { return _outputInfo; }

    std::pair<const char*, size_t> getHeader() const { return {_pBlob, sizeof(ElfN_Ehdr) + sizeof(mv_blob_header)};}

private:
    const char* _pBlob = nullptr;

    mv_blob_header _blobHeader = {};

    ie::InputsDataMap  _networkInputs;
    ie::OutputsDataMap _networkOutputs;

    DataInfo _inputInfo;
    DataInfo _outputInfo;
};

}  // namespace vpu

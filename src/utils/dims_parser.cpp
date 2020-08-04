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

#include <details/ie_exception.hpp>
#include <dims_parser.hpp>

namespace vpu {

void parseDims(const InferenceEngine::TensorDesc& desc, size_t& dimN, size_t& dimZ, size_t& dimY, size_t& dimX,
    size_t defaultValue) {
    dimN = dimZ = dimY = dimX = defaultValue;
    const auto& dims = desc.getDims();
    switch (dims.size()) {
    case 1:
        dimZ = dims[0];
        break;
    case 2:
        dimN = dims[0];
        dimZ = dims[1];
        break;
    case 3:
        dimX = dims[2];
        dimY = dims[1];
        dimZ = dims[0];
        break;
    case 4:
        dimX = dims[3];
        dimY = dims[2];
        dimZ = dims[1];
        dimN = dims[0];
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported dimensions layout";
        break;
    }
}

InferenceEngine::TensorDesc getNCHW(const InferenceEngine::TensorDesc& desc, size_t defaultValue) {
    size_t dimN, dimZ, dimY, dimX;
    parseDims(desc, dimN, dimZ, dimY, dimX, defaultValue);
    InferenceEngine::TensorDesc new_desc(desc.getPrecision(), {dimN, dimZ, dimY, dimX}, InferenceEngine::Layout::NCHW);
    return new_desc;
}

InferenceEngine::TensorDesc getWHCN(const InferenceEngine::TensorDesc& desc, size_t defaultValue) {
    size_t dimN, dimZ, dimY, dimX;
    parseDims(desc, dimN, dimZ, dimY, dimX, defaultValue);
    InferenceEngine::TensorDesc new_desc(desc.getPrecision(), {dimX, dimY, dimZ, dimN}, InferenceEngine::Layout::ANY);
    return new_desc;
}

}  // namespace vpu
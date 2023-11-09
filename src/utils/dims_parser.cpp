//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <dims_parser.hpp>

namespace vpu {

void parseDims(const InferenceEngine::SizeVector& dims, size_t& dimN, size_t& dimZ, size_t& dimY, size_t& dimX,
               size_t& dimD, size_t defaultValue) {
    dimN = dimZ = dimY = dimX = dimD = defaultValue;
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
    case 5:
        dimX = dims[4];
        dimY = dims[3];
        dimD = dims[2];
        dimZ = dims[1];
        dimN = dims[0];
        break;
    default:
        IE_THROW() << "Unsupported dimensions layout";
        break;
    }
}

InferenceEngine::TensorDesc getNCHW(const InferenceEngine::TensorDesc& desc, size_t defaultValue) {
    size_t dimN, dimZ, dimY, dimX, dimD;
    parseDims(desc.getDims(), dimN, dimZ, dimY, dimX, dimD, defaultValue);
    InferenceEngine::TensorDesc new_desc;
    if (desc.getDims().size() == 5) {
        new_desc = InferenceEngine::TensorDesc(desc.getPrecision(), {dimN, dimZ, dimD, dimY, dimX},
                                               InferenceEngine::Layout::NCDHW);
    } else {
        new_desc = InferenceEngine::TensorDesc(desc.getPrecision(), {dimN, dimZ, dimY, dimX},
                                               InferenceEngine::Layout::NCHW);
    }
    return new_desc;
}

InferenceEngine::TensorDesc getWHCN(const InferenceEngine::TensorDesc& desc, size_t defaultValue) {
    size_t dimN, dimZ, dimY, dimX, dimD;
    parseDims(desc.getDims(), dimN, dimZ, dimY, dimX, dimD, defaultValue);
    InferenceEngine::TensorDesc new_desc;
    if (desc.getDims().size() == 5) {
        new_desc = InferenceEngine::TensorDesc(desc.getPrecision(), {dimX, dimY, dimD, dimZ, dimN},
                                               InferenceEngine::Layout::ANY);
    } else {
        new_desc = InferenceEngine::TensorDesc(desc.getPrecision(), {dimX, dimY, dimZ, dimN},
                                               InferenceEngine::Layout::ANY);
    }
    return new_desc;
}

}  // namespace vpu

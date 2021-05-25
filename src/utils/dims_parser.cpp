//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

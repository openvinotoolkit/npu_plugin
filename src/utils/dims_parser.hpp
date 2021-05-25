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

#pragma once

#include <ie_layouts.h>

namespace vpu {

void parseDims(const InferenceEngine::SizeVector& dims, size_t& dimN, size_t& dimZ, size_t& dimY, size_t& dimX,
               size_t& dimD, size_t defaultValue = 1);

InferenceEngine::TensorDesc getNCHW(const InferenceEngine::TensorDesc& desc, size_t defaultValue = 1);

InferenceEngine::TensorDesc getWHCN(const InferenceEngine::TensorDesc& desc, size_t defaultValue = 1);

}  // namespace vpu

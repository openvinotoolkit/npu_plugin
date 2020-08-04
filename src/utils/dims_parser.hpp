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

#include <ie_layouts.h>

namespace vpu {

void parseDims(const InferenceEngine::TensorDesc& desc, size_t& dimN, size_t& dimZ, size_t& dimY, size_t& dimX,
    size_t defaultValue = 1);

InferenceEngine::TensorDesc getNCHW(const InferenceEngine::TensorDesc& desc, size_t defaultValue = 1);

InferenceEngine::TensorDesc getWHCN(const InferenceEngine::TensorDesc& desc, size_t defaultValue = 1);

}  // namespace vpu
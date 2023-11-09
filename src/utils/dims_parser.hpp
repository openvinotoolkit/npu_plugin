//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ie_layouts.h>

namespace vpu {

void parseDims(const InferenceEngine::SizeVector& dims, size_t& dimN, size_t& dimZ, size_t& dimY, size_t& dimX,
               size_t& dimD, size_t defaultValue = 1);

InferenceEngine::TensorDesc getNCHW(const InferenceEngine::TensorDesc& desc, size_t defaultValue = 1);

InferenceEngine::TensorDesc getWHCN(const InferenceEngine::TensorDesc& desc, size_t defaultValue = 1);

}  // namespace vpu

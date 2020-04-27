//
// Copyright 2020 Intel Corporation.
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

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile/graphfile_generated.h>

InferenceEngine::Layout orderToLayout(const std::vector<uint32_t>& tensorOrder);
InferenceEngine::Precision DTypeToPrecision(const MVCNN::DType& dtype);

#ifdef ENABLE_MCM_COMPILER
std::vector<uint32_t> layoutToOrder(const InferenceEngine::Layout& tensorLayout);
MVCNN::DType precisionToDType(const InferenceEngine::Precision& tensorPrecision);
#endif

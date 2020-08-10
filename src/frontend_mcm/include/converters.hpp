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
#include <ie_common.h>
#include <schema/graphfile/graphfile_generated.h>

#include <ie_precision.hpp>

InferenceEngine::Layout orderVectorToLayout(const std::vector<uint32_t>& tensorOrder);
InferenceEngine::Precision MvcnnDTypeToPrecision(const MVCNN::DType& dtype);

#ifdef ENABLE_MCM_COMPILER

#include <mcm/tensor/dtype/dtype.hpp>
#include <mcm/tensor/order/order.hpp>
#include <mcm/tensor/shape.hpp>

std::vector<uint32_t> layoutToOrderVector(const InferenceEngine::Layout& tensorLayout);
MVCNN::DType precisionToMvcnnDType(const InferenceEngine::Precision& tensorPrecision);
mv::DType precisionToDType(const InferenceEngine::Precision& iePrecision);
mv::Order layoutToOrder(const InferenceEngine::Layout& ieLayout);
mv::Shape sizeVectorToShape(InferenceEngine::SizeVector dims);

#endif

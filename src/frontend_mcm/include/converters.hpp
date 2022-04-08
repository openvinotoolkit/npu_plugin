//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_private_config.hpp"

#include <schema/graphfile/graphfile_generated.h>

#include <ie_common.h>
#include <ie_precision.hpp>

InferenceEngine::Layout getLayout(const MVCNN::TensorReference* const tensorRef);
InferenceEngine::Precision MvcnnDTypeToPrecision(const MVCNN::DType& dtype);

#include <mcm/tensor/dtype/dtype.hpp>
#include <mcm/tensor/order/order.hpp>
#include <mcm/tensor/shape.hpp>

std::vector<float> layoutToOrderVector(const InferenceEngine::Layout& tensorLayout);
MVCNN::DType precisionToMvcnnDType(const InferenceEngine::Precision& tensorPrecision);
mv::DType precisionToDType(const InferenceEngine::Precision& iePrecision);
mv::Order layoutToOrder(const InferenceEngine::Layout& ieLayout);
mv::Shape sizeVectorToShape(InferenceEngine::SizeVector dims);
MVCNN::TargetDeviceRevision getDeviceRevision(const InferenceEngine::VPUXConfigParams::VPUXPlatform platform);

//
// Copyright 2020 Intel Corporation.
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

#include <flatbuffers/flatbuffers.h>
#include <ie_common.h>
#include <schema/graphfile/graphfile_generated.h>
#include <ie_precision.hpp>
#include <vpux_config.hpp>

InferenceEngine::Layout orderVectorToLayout(const std::vector<float>& tensorOrder);
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

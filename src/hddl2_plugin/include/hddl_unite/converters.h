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

#include "ie_precision.hpp"

namespace vpu {
namespace HDDL2Plugin {
namespace Unite {

HddlUnite::Inference::Precision convertFromIEPrecision(const InferenceEngine::Precision& precision) {
    switch (precision) {
    case InferenceEngine::Precision::UNSPECIFIED:
        return HddlUnite::Inference::UNSPECIFIED;
    case InferenceEngine::Precision::MIXED:
        return HddlUnite::Inference::MIXED;
    case InferenceEngine::Precision::FP32:
        return HddlUnite::Inference::FP32;
    case InferenceEngine::Precision::FP16:
        return HddlUnite::Inference::FP16;
    case InferenceEngine::Precision::Q78:
        return HddlUnite::Inference::Q78;
    case InferenceEngine::Precision::I16:
        return HddlUnite::Inference::I16;
    case InferenceEngine::Precision::U8:
        return HddlUnite::Inference::U8;
    case InferenceEngine::Precision::I8:
        return HddlUnite::Inference::I8;
    case InferenceEngine::Precision::U16:
        return HddlUnite::Inference::U16;
    case InferenceEngine::Precision::I32:
        return HddlUnite::Inference::I32;
    case InferenceEngine::Precision::I64:
        return HddlUnite::Inference::I64;
    case InferenceEngine::Precision::BIN:
        return HddlUnite::Inference::BIN;
    case InferenceEngine::Precision::CUSTOM:
        return HddlUnite::Inference::CUSTOM;
    default:
        THROW_IE_EXCEPTION << "Incorrect precision";
    }
}

}  // namespace Unite
}  // namespace HDDL2Plugin
}  // namespace vpu

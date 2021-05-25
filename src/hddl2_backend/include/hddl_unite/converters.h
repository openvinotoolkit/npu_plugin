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

// IE
#include "ie_precision.hpp"
// Plugin
#include <vpux_config.hpp>
// Low-level
#include <HddlUnite.h>
#include <Inference.h>

namespace vpux {
namespace hddl2 {
namespace Unite {

inline HddlUnite::Inference::Precision convertFromIEPrecision(const InferenceEngine::Precision& precision) {
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
        IE_THROW() << "Incorrect precision";
    }
}

inline HddlUnite::clientLogLevel convertIELogLevelToUnite(const vpu::LogLevel ieLogLevel) {
    switch (ieLogLevel) {
    case vpu::LogLevel::None:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    case vpu::LogLevel::Fatal:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    case vpu::LogLevel::Error:
        return HddlUnite::clientLogLevel::LOGLEVEL_ERROR;
    case vpu::LogLevel::Warning:
        return HddlUnite::clientLogLevel::LOGLEVEL_WARN;
    case vpu::LogLevel::Info:
        return HddlUnite::clientLogLevel::LOGLEVEL_INFO;
    case vpu::LogLevel::Debug:
        return HddlUnite::clientLogLevel::LOGLEVEL_DEBUG;
    case vpu::LogLevel::Trace:
        return HddlUnite::clientLogLevel::LOGLEVEL_PROCESS;
    default:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    }
}

}  // namespace Unite
}  // namespace hddl2
}  // namespace vpux

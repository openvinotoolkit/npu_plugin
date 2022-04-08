//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/logger.hpp"

// IE
#include <ie_precision.hpp>

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

inline HddlUnite::clientLogLevel convertIELogLevelToUnite(const LogLevel ieLogLevel) {
    switch (ieLogLevel) {
    case LogLevel::None:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    case LogLevel::Fatal:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    case LogLevel::Error:
        return HddlUnite::clientLogLevel::LOGLEVEL_ERROR;
    case LogLevel::Warning:
        return HddlUnite::clientLogLevel::LOGLEVEL_WARN;
    case LogLevel::Info:
        return HddlUnite::clientLogLevel::LOGLEVEL_INFO;
    case LogLevel::Debug:
        return HddlUnite::clientLogLevel::LOGLEVEL_DEBUG;
    case LogLevel::Trace:
        return HddlUnite::clientLogLevel::LOGLEVEL_PROCESS;
    default:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    }
}

}  // namespace Unite
}  // namespace hddl2
}  // namespace vpux

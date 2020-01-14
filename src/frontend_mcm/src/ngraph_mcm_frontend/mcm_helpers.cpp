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

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <details/ie_exception.hpp>
#include <limits>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

std::string cvtLogLevelToMCM(vpu::LogLevel lvl) {
    switch (lvl) {
    case vpu::LogLevel::None:
        return "Silent";

    case vpu::LogLevel::Fatal:
    case vpu::LogLevel::Error:
        return "Error";

    case vpu::LogLevel::Warning:
        return "Warning";

    case vpu::LogLevel::Info:
        return "Info";

    case vpu::LogLevel::Debug:
    case vpu::LogLevel::Trace:
        return "Debug";

    default:
        return "Silent";
    }
}

mv::QuantizationParams makeQuantParams() {
    static const auto INITIAL_QUANT_PARAMS = makeQuantParams({0}, {1.0});
    return INITIAL_QUANT_PARAMS;
}

mv::QuantizationParams makeQuantParams(const std::vector<int64_t>& zeroPoints, const std::vector<double>& scales) {
    IE_ASSERT(zeroPoints.size() == scales.size());
    constexpr double INF_FP64 = std::numeric_limits<double>::infinity();
    return mv::QuantizationParams(zeroPoints, scales, {-INF_FP64}, {INF_FP64});
}

mv::Shape cvtShapeToMCM(const ngraph::Shape& shape) {
    std::vector<size_t> dims = shape;
    std::reverse(dims.begin(), dims.end());
    return mv::Shape(dims);
}

mv::Shape cvtShapeToMCM(const ngraph::PartialShape& pshape) {
    return cvtShapeToMCM(pshape.to_shape());
}

mv::DType cvtElemTypeToMCM(const ngraph::element::Type& elemType) {
    std::ostringstream mcmTypeName;

    if (elemType.is_real()) {
        mcmTypeName << "Float";
    } else if (elemType.is_signed()) {
        mcmTypeName << "Int";
    } else {
        mcmTypeName << "UInt";
    }

    mcmTypeName << elemType.bitwidth();

    return mv::DType(mcmTypeName.str());
}

#endif
// clang-format on

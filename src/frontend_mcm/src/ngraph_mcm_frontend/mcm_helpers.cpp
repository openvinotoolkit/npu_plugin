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

#include "ngraph_mcm_frontend/mcm_helpers.hpp"

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

mv::DType cvtOutputType(const ngraph::element::Type& elemType) {
    if (ngraph::element::f32 == elemType)
        return mv::DType("Float16");
    else if (ngraph::element::f16 == elemType)
        return mv::DType("Float16");
    else if (ngraph::element::u8 == elemType)
        return mv::DType("UInt8");
    else {
        std::stringstream msg;
        msg << "Unsupported output element type: " << elemType;
        IE_ASSERT(msg.str().c_str());
        return mv::DType("Default");
    }
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
        if (elemType == ngraph::element::bf16)
            mcmTypeName << "B";
        mcmTypeName << "Float";
    } else if (elemType.is_signed()) {
        mcmTypeName << "Int";
    } else {
        mcmTypeName << "UInt";
    }
    mcmTypeName << elemType.bitwidth();

    return mv::DType(mcmTypeName.str());
}

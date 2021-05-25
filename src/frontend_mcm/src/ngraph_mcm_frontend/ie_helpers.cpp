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

// clang-format off

#include "ngraph_mcm_frontend/ie_helpers.hpp"

ngraph::element::Type cvtPrecisionToElemType(const ie::Precision& precision) {
    switch (precision) {
    case ie::Precision::FP32:
        return ngraph::element::f32;
    case ie::Precision::FP16:
        return ngraph::element::f16;
    case ie::Precision::I64:
        return ngraph::element::i64;
    case ie::Precision::I32:
        return ngraph::element::i32;
    case ie::Precision::I16:
        return ngraph::element::i16;
    case ie::Precision::I8:
        return ngraph::element::i8;
    case ie::Precision::U16:
        return ngraph::element::u16;
    case ie::Precision::U8:
        return ngraph::element::u8;
    case ie::Precision::BF16:
        return ngraph::element::bf16;
    default:
        IE_THROW() << "Unsupported precision " << precision;
    }
}

// clang-format on

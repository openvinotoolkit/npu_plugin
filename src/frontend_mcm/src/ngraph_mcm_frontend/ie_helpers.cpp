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

#include "ngraph_mcm_frontend/ie_helpers.hpp"
#include <details/ie_exception.hpp>

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
    default:
        THROW_IE_EXCEPTION << "Unsupported precision " << precision;
    }
}

#endif
// clang-format on

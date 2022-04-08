//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/format.hpp"

#include <ie_blob.h>
#include <ie_data.h>

#include <ngraph/partial_shape.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/strides.hpp>
#include <ngraph/type/element_type.hpp>

#include <sstream>

//
// Format
//

namespace llvm {

template <>
struct format_provider<InferenceEngine::Layout> final {
    static void format(const InferenceEngine::Layout& layout, llvm::raw_ostream& stream, StringRef style) {
        std::ostringstream strm;
        strm << layout;

        llvm::detail::build_format_adapter(strm.str()).format(stream, style);
    }
};

template <>
struct format_provider<InferenceEngine::Precision> final {
    static void format(const InferenceEngine::Precision& precision, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(precision.name()).format(stream, style);
    }
};

template <>
struct format_provider<InferenceEngine::TensorDesc> final {
    static void format(const InferenceEngine::TensorDesc& ieTensorDesc, llvm::raw_ostream& stream, StringRef style) {
        stream << '<';
        llvm::detail::build_format_adapter(ieTensorDesc.getPrecision()).format(stream, style);
        stream << ", ";
        llvm::detail::build_format_adapter(ieTensorDesc.getDims()).format(stream, style);
        stream << ", ";
        llvm::detail::build_format_adapter(ieTensorDesc.getLayout()).format(stream, style);
        stream << '>';
    }
};

template <>
struct format_provider<InferenceEngine::Data> final {
    static void format(const InferenceEngine::Data& ieData, llvm::raw_ostream& stream, StringRef style) {
        stream << "<\"";
        llvm::detail::build_format_adapter(ieData.getName()).format(stream, style);
        stream << "\", ";
        llvm::detail::build_format_adapter(ieData.getTensorDesc()).format(stream, style);
        stream << '>';
    }
};

template <>
struct format_provider<InferenceEngine::Blob> final {
    static void format(const InferenceEngine::Blob& ieBlob, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(ieBlob.getTensorDesc()).format(stream, style);
    }
};

template <>
struct format_provider<ngraph::element::Type> final {
    static void format(const ngraph::element::Type& elemType, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(elemType.get_type_name()).format(stream, style);
    }
};

template <>
struct format_provider<ngraph::Shape> final : vpux::ListFormatProvider {};

template <>
struct format_provider<ngraph::Strides> final : vpux::ListFormatProvider {};

template <>
struct format_provider<ngraph::PartialShape> {
    static void format(const ngraph::PartialShape& pshape, llvm::raw_ostream& stream, StringRef style) {
        std::ostringstream strm;
        strm << pshape;

        llvm::detail::build_format_adapter(strm.str()).format(stream, style);
    }
};

}  // namespace llvm

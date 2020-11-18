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

#include "vpux/utils/core/format.hpp"

#include <ie_blob.h>
#include <ie_data.h>

#include <ngraph/partial_shape.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include <sstream>

//
// Format
//

namespace llvm {

template <>
struct format_provider<InferenceEngine::Layout> final {
    static void format(const InferenceEngine::Layout& layout,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        std::ostringstream strm;
        strm << layout;

        auto adapter = llvm::detail::build_format_adapter(strm.str());
        adapter.format(stream, style);
    }
};

template <>
struct format_provider<InferenceEngine::Precision> final {
    static void format(const InferenceEngine::Precision& precision,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        auto adapter = llvm::detail::build_format_adapter(precision.name());
        adapter.format(stream, style);
    }
};

template <>
struct format_provider<InferenceEngine::TensorDesc> final {
    static void format(const InferenceEngine::TensorDesc& ieTensorDesc,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        stream << '<';

        auto adapter1 =
                llvm::detail::build_format_adapter(ieTensorDesc.getPrecision());
        adapter1.format(stream, style);

        stream << ", ";

        auto adapter2 =
                llvm::detail::build_format_adapter(ieTensorDesc.getDims());
        adapter2.format(stream, style);

        stream << ", ";

        auto adapter3 =
                llvm::detail::build_format_adapter(ieTensorDesc.getLayout());
        adapter3.format(stream, style);

        stream << '>';
    }
};

template <>
struct format_provider<InferenceEngine::Data> final {
    static void format(const InferenceEngine::Data& ieData,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        stream << "<\"";

        auto adapter1 = llvm::detail::build_format_adapter(ieData.getName());
        adapter1.format(stream, style);

        stream << "\", ";

        auto adapter2 =
                llvm::detail::build_format_adapter(ieData.getTensorDesc());
        adapter2.format(stream, style);

        stream << '>';
    }
};

template <>
struct format_provider<InferenceEngine::Blob> final {
    static void format(const InferenceEngine::Blob& ieBlob,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        auto adapter =
                llvm::detail::build_format_adapter(ieBlob.getTensorDesc());
        adapter.format(stream, style);
    }
};

template <>
struct format_provider<ngraph::element::Type> final {
    static void format(const ngraph::element::Type& elemType,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        auto adapter =
                llvm::detail::build_format_adapter(elemType.get_type_name());
        adapter.format(stream, style);
    }
};

template <>
struct format_provider<ngraph::Shape> final : vpux::ContainerFormatter {};

template <>
struct format_provider<ngraph::PartialShape> {
    static void format(const ngraph::PartialShape& pshape,
                       llvm::raw_ostream& stream,
                       StringRef style) {
        std::ostringstream strm;
        strm << pshape;

        auto adapter = llvm::detail::build_format_adapter(strm.str());
        adapter.format(stream, style);
    }
};

}  // namespace llvm

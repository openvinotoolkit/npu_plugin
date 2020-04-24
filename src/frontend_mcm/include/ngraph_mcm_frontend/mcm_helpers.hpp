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

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include <vpu/utils/logger.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/node.hpp>
#include <include/mcm/tensor/quantization_params.hpp>
#include <include/mcm/tensor/shape.hpp>
#include <include/mcm/tensor/dtype/dtype.hpp>
#include <include/mcm/computation/model/iterator/tensor.hpp>
#include <string>
#include <vector>
#include <unordered_map>

std::string cvtLogLevelToMCM(vpu::LogLevel lvl);

mv::QuantizationParams makeQuantParams();
mv::QuantizationParams makeQuantParams(const std::vector<int64_t>& zeroPoints, const std::vector<double>& scales);

mv::Shape cvtShapeToMCM(const ngraph::Shape& shape);
mv::Shape cvtShapeToMCM(const ngraph::PartialShape& pshape);

mv::DType cvtElemTypeToMCM(const ngraph::element::Type& elemType);
mv::DType cvtOutputType(const ngraph::element::Type& elemType);

struct NodeOutputHash final {
    static size_t hash_combine(size_t seed, size_t val) {
        // Hash combine formula from boost
        return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    size_t operator()(const ngraph::Output<ngraph::Node>& out) const {
        return hash_combine(reinterpret_cast<uintptr_t>(out.get_node()), out.get_index());
    }
};

using NodeOutputToMcmMap = std::unordered_map<ngraph::Output<ngraph::Node>, mv::Data::TensorIterator, NodeOutputHash>;

#endif
// clang-format on

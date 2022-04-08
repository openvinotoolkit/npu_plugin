//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <include/mcm/computation/model/iterator/tensor.hpp>
#include <include/mcm/tensor/dtype/dtype.hpp>
#include <include/mcm/tensor/quantization_params.hpp>
#include <include/mcm/tensor/shape.hpp>
#include <ngraph/node.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <string>
#include <unordered_map>
#include <vector>

std::string cvtLogLevelToMCM(vpux::LogLevel lvl);

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

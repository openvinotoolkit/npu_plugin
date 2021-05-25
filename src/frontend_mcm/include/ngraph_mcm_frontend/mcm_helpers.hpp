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
#include <vpu/utils/logger.hpp>

std::string cvtLogLevelToMCM(vpu::LogLevel lvl);

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

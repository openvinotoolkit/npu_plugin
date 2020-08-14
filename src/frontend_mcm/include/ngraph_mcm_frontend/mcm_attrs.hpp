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

#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <ie_layouts.h>
#include <ngraph/node.hpp>
#include <ngraph/descriptor/tensor.hpp>
#include <ngraph/op/util/op_annotations.hpp>
#include <include/mcm/tensor/order/order.hpp>
#include <include/mcm/tensor/quantization_params.hpp>
#include <memory>
#include <unordered_map>
#include <algorithm>

namespace ie = InferenceEngine;

mv::Order cvtLayoutToMCM(ie::Layout layout);

class McmOpAttrs final : public ngraph::op::util::OpAnnotations {
public:
    static McmOpAttrs& get(std::shared_ptr<ngraph::Node> node);
    static void copy(std::shared_ptr<ngraph::Node> src_node, std::shared_ptr<ngraph::Node> dst_node);

    static const mv::QuantizationParams& getQuantParams(std::shared_ptr<ngraph::Node> node);
    static void setQuantParams(const mv::QuantizationParams& quantParams, std::shared_ptr<ngraph::Node> node);

    static const mv::Order& getOrder(std::shared_ptr<ngraph::Node> node, size_t outInd = 0);
    static void setOrder(const mv::Order& order, std::shared_ptr<ngraph::Node> node, size_t outInd = 0);

private:
    mv::QuantizationParams _mvQuantParams = makeQuantParams();
    std::unordered_map<size_t, mv::Order> _mvOrders;
};

#endif
// clang-format on

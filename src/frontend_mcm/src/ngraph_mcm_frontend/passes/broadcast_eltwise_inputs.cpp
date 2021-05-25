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

#include "ngraph_mcm_frontend/passes/broadcast_eltwise_inputs.hpp"

#include <vector>
#include <memory>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/minimum.hpp>
#include <ngraph/op/constant.hpp>
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/attr_types.hpp"

bool BroadcastEltwiseInputs::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (std::dynamic_pointer_cast<ngraph::op::v1::Maximum>(node) != nullptr ||
    std::dynamic_pointer_cast<ngraph::op::v1::Minimum>(node) != nullptr ) {
        ngraph::PartialShape outShape = node->get_output_partial_shape(0);
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            ngraph::PartialShape pShape = node->get_input_partial_shape(i);
            const auto input = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node->input_value(i).get_node_shared_ptr());
            if ((pShape != outShape) && (input != nullptr)) {
                auto oldValues = input->cast_vector<double>();
                if (oldValues.size() == 1) {
                    // merge shape and broadcast
                    if (!ngraph::PartialShape::broadcast_merge_into(pShape, outShape, node->get_autob())) {
                        return false;
                    }
                    auto newConstInput = std::make_shared<ngraph::op::v0::Constant>(
                                            input->get_element_type(),
                                            pShape.to_shape(),
                                            oldValues[0]);
                    ngraph::replace_node(input, newConstInput);
                } else {
                    // can broadcast only scalar value
                    return false;
                }
            }
        }
    }
    return true;
}

// clang-format on

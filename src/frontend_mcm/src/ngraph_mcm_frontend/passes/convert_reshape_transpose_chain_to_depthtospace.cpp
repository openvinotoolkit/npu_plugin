//
// Copyright Intel Corporation.
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

// clang-format on

#include "ngraph_mcm_frontend/passes/convert_reshape_transpose_chain_to_depthtospace.hpp"

#include <memory>
#include <ngraph/op/depth_to_space.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/transpose.hpp>

// it's a workaround for edsr3. check details on ticket EISW-16823
bool ConvertReshapeTransposeChainToDepthToSpace::run_on_node(std::shared_ptr<ngraph::Node> node) {
    if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node) != nullptr) {
        auto node_1 = node->input_value(0).get_node_shared_ptr();
        if (std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node_1) != nullptr) {
            auto node_2 = node_1->input_value(0).get_node_shared_ptr();
            if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node_2) != nullptr) {
                auto node_3 = node_2->input_value(0).get_node_shared_ptr();
                if (std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node_3) != nullptr) {
                    auto node_4 = node_3->input_value(0).get_node_shared_ptr();
                    if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node_4) != nullptr) {
                        auto node_5 = node_4->input_value(0).get_node_shared_ptr();
                        if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node_5) != nullptr) {
                            auto node_6 = node_5->input_value(0).get_node_shared_ptr();
                            if (std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node_6) != nullptr) {
                                auto node_7 = node_6->input_value(0).get_node_shared_ptr();
                                if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node_7) != nullptr) {
                                    auto src_shape = node_7->get_input_shape(0);
                                    auto dst_shape = node->get_output_shape(0);
                                    if ((src_shape.size() == dst_shape.size()) && (src_shape[1] == 4 * dst_shape[1]) &&
                                        (dst_shape[2] == 2 * src_shape[2]) && (dst_shape[3] == 2 * src_shape[3])) {
                                        const auto depthToSpace = std::make_shared<ngraph::op::v0::DepthToSpace>(
                                                node_7->input_value(0), "DEPTH_FIRST", 2);
                                        ngraph::replace_node(node, depthToSpace);
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return false;
}

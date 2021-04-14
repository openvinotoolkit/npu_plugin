// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_with_offset_op.hpp"

using namespace SampleExtension;

constexpr ngraph::NodeTypeInfo AddWOffsetOp::type_info;

AddWOffsetOp::AddWOffsetOp(const ngraph::Output<ngraph::Node>& inp1, const ngraph::Output<ngraph::Node>& inp2, float _offset)
    : Op({inp1, inp2}) {
    constructor_validate_and_infer_types();
    offset = _offset;
}

void AddWOffsetOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}

std::shared_ptr<ngraph::Node> AddWOffsetOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 2) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<AddWOffsetOp>(new_args.at(0), new_args.at(1), offset);
}

bool AddWOffsetOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("offset", offset);
    return true;
}


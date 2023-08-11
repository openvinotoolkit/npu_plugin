//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/opsets/opset10.hpp>

#include <memory>

class EmptyNode {
public:
    static const ov::Node* instance() {
        static std::unique_ptr<ov::Node> emptyNode = []() {
            auto node = std::make_unique<ov::opset10::Constant>(ov::element::Type_t::f32, ov::Shape{1});
            node->set_friendly_name("EmptyNode");
            return node;
        }();

        return emptyNode.get();
    }
};

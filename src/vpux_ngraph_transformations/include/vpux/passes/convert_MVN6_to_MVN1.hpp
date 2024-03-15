//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>

namespace vpux {

namespace passes {

class ConvertMVN6toMVN1 : public ov::pass::MatcherPass {
public:
    ConvertMVN6toMVN1();
};

}  // namespace passes
}  // namespace vpux

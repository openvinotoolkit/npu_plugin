//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>

namespace vpux {

namespace passes {

class ConvertLayerNormToMVN : public ov::pass::MatcherPass {
public:
    ConvertLayerNormToMVN();
};

class ConvertInstanceNormToMVN : public ov::pass::MatcherPass {
public:
    ConvertInstanceNormToMVN();
};

}  // namespace passes
}  // namespace vpux

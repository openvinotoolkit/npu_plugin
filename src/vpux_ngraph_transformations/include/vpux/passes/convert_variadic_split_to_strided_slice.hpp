//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>

namespace vpux {

namespace passes {

class ConvertVariadicSplitToStridedSliceOp : public ov::pass::MatcherPass {
public:
    ConvertVariadicSplitToStridedSliceOp();
};

}  // namespace passes
}  // namespace vpux

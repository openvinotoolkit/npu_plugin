//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>

namespace vpux {
namespace passes {

class OnnxReorgPatternToDarkNetReorg : public ngraph::pass::MatcherPass {
public:
    OnnxReorgPatternToDarkNetReorg();
};

}  // namespace passes
}  // namespace vpux

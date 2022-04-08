//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>

namespace vpux {

namespace passes {


class ConvertMVN6toMVN1 : public ngraph::pass::MatcherPass {
public:
    ConvertMVN6toMVN1();
};

}  // namespace passes
}  // namespace vpux
// clang-format on

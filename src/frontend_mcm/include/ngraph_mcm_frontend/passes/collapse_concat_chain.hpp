//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class CollapseConcats0238 : public MatcherPass {
public:
    CollapseConcats0238();
};

}  // namespace pass
}  // namespace ngraph

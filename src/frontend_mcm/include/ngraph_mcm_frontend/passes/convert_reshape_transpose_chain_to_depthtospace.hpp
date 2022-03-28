//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>

class ConvertReshapeTransposeChainToDepthToSpace : public ngraph::pass::MatcherPass {
public:
    ConvertReshapeTransposeChainToDepthToSpace();
};
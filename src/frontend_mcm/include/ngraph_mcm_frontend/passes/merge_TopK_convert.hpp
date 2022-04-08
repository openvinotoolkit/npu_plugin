//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ngraph/pass/graph_rewrite.hpp>

//
// Merge [Topk] -> [Convert] -> [Result] into [TopK] -> [Result].
//

class MergeTopKConvert : public ngraph::pass::FunctionPass  {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

// clang-format on

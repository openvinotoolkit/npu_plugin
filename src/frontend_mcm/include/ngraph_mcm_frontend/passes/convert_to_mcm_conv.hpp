//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ngraph/pass/pass.hpp>
#include <memory>

//
// Convert [ConvolutionIE] to [McmConv] with optional [McmBias].
//

class ConvertToMcmConv final : public ngraph::pass::NodePass {
public:
    bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
};

// clang-format on

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <ngraph/pass/pass.hpp>

class DetectInputFQ final : public ngraph::pass::FunctionPass {
public:
    DetectInputFQ(bool* needConvertInputPrecision): _needConvertInputPrecision(needConvertInputPrecision) {
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    bool* _needConvertInputPrecision = nullptr;
};

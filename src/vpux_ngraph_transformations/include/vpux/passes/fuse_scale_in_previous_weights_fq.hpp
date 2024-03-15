//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/pass/pass.hpp>

namespace vpux {
namespace pass {

class FuseScaleAfterClamp final : public ov::pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

}  // namespace pass
}  // namespace vpux

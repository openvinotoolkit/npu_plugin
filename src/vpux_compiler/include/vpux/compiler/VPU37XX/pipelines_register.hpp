//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/pipelines_register.hpp"

namespace vpux {

//
// PipelineRegister37XX
//

class PipelineRegister37XX final : public IPipelineRegister {
public:
    void registerPipelines() override;
};

}  // namespace vpux

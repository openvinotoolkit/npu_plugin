//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/pipelines_register.hpp"
#include "vpux/compiler/VPU37XX/pipelines_register.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// createPipelineRegister
//

std::unique_ptr<IPipelineRegister> vpux::createPipelineRegister(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return std::make_unique<PipelineRegister30XX>();
    case VPU::ArchKind::VPUX37XX:
        return std::make_unique<PipelineRegister37XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

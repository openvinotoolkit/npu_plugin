//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// This file is shared between the compiler and profiling post-processing

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/profiling.hpp"

namespace vpux::profiling {

std::string convertExecTypeToName(ExecutorType execType) {
    switch (execType) {
    case ExecutorType::ACTSHAVE:
        return "actshave";
    case ExecutorType::DMA_HW:
        return "dmahw";
    case ExecutorType::DMA_SW:
        return "dma";
    case ExecutorType::DPU:
        return "dpu";
    case ExecutorType::UPA:
        return "upa";
    case ExecutorType::WORKPOINT:
        return "pll";
    default:
        VPUX_THROW("Unknown execType");
    };
}

ExecutorType convertDataInfoNameToExecType(StringRef name) {
    if (name == "actshave") {
        return ExecutorType::ACTSHAVE;
    } else if (name == "dmahw") {
        return ExecutorType::DMA_HW;
    } else if (name == "dma") {
        return ExecutorType::DMA_SW;
    } else if (name == "dpu") {
        return ExecutorType::DPU;
    } else if (name == "upa") {
        return ExecutorType::UPA;
    } else if (name == "pll") {
        return ExecutorType::WORKPOINT;
    }
    VPUX_THROW("Can not convert '{0}' to profiling::ExecutorType", name);
}

}  // namespace vpux::profiling

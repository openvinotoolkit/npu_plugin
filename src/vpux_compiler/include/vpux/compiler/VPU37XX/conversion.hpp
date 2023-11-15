//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"

namespace vpux {
namespace arch37xx {

//
// PermuteQuant options
//

struct PermuteQuantOptions : mlir::PassPipelineOptions<PermuteQuantOptions> {
    BoolOption useNCEPermute{*this, "use-nce-permute", llvm::cl::desc("Use nce permute operation"),
                             llvm::cl::init(false)};

    PermuteQuantOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit PermuteQuantOptions(const OtherOptions& options) {
        useNCEPermute = options.useNCEPermute;
    }
};

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertIEToVPUNCEPass(bool useNCEPermute = false, Logger log = Logger::global());

//
// Pipelines
//

void buildLowerIE2VPUPipeline37XX(mlir::OpPassManager& pm, const PermuteQuantOptions& options,
                                  Logger log = Logger::global());
void buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowerVPU2VPUIP37XXPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// registerConversionPipeline37XX
//

void registerConversionPipeline37XX();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU37XX/conversion/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU37XX/conversion/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch37xx
}  // namespace vpux

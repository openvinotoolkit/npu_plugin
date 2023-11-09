//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/pipelines_register.hpp"
#include "vpux/compiler/VPU37XX/conversion.hpp"
#include "vpux/compiler/VPU37XX/dialect/IE/passes.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPU/passes.hpp"
#include "vpux/compiler/VPU37XX/pipelines.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry37XX::registerPipelines
//

void PipelineRegistry37XX::registerPipelines() {
    mlir::PassPipelineRegistration<>("ShaveCodeGen", "Compile both from IE to VPUIP and from IERT to LLVM for VPU37XX",
                                     [](mlir::OpPassManager& pm) {
                                         buildShaveCodeGenPipeline37XX(pm);
                                     });

    mlir::PassPipelineRegistration<ReferenceSWOptions37XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for VPU37XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions37XX& options) {
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX37XX, VPU::CompilationMode::ReferenceSW));

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceHWOptions37XX>(
            "reference-hw-mode", "Compile IE Network in Reference Hardware mode (HW and SW execution) for VPU37XX",
            [](mlir::OpPassManager& pm, const ReferenceHWOptions37XX& options) {
                const auto numOfDPUGroups = convertToOptional(options.numberOfDPUGroups);
                const auto numOfDMAPorts = convertToOptional(options.numberOfDMAPorts);
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX37XX, VPU::CompilationMode::ReferenceHW,
                                                       numOfDPUGroups, numOfDMAPorts));

                buildReferenceHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions37XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for VPU37XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions37XX& options) {
                const auto numOfDPUGroups = convertToOptional(options.numberOfDPUGroups);
                const auto numOfDMAPorts = convertToOptional(options.numberOfDMAPorts);
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX37XX, VPU::CompilationMode::DefaultHW,
                                                       numOfDPUGroups, numOfDMAPorts));

                buildDefaultHWModePipeline(pm, options);
            });
    vpux::IE::arch37xx::registerIEPipelines();
    vpux::VPU::arch37xx::registerVPUPipelines();
    vpux::arch37xx::registerConversionPipeline37XX();
}

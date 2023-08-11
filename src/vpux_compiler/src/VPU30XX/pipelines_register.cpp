//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/pipelines_register.hpp"
#include "vpux/compiler/VPU30XX/pipelines.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegister30XX::registerPipelines
//

void PipelineRegister30XX::registerPipelines() {
    mlir::PassPipelineRegistration<>("ShaveCodeGen", "Compile both from IE to VPUIP and from IERT to LLVM for VPU30XX",
                                     [](mlir::OpPassManager& pm) {
                                         buildShaveCodeGenPipeline30XX(pm);
                                     });

    mlir::PassPipelineRegistration<ReferenceSWOptions30XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for VPU30XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options) {
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX30XX, VPU::CompilationMode::ReferenceSW));

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceHWOptions30XX>(
            "reference-hw-mode", "Compile IE Network in Reference Hardware mode (HW and SW execution) for VPU30XX",
            [](mlir::OpPassManager& pm, const ReferenceHWOptions30XX& options) {
                const auto numOfDPUGroups = convertToOptional(options.numberOfDPUGroups);
                const auto numOfDMAPorts = convertToOptional(options.numberOfDMAPorts);
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX30XX, VPU::CompilationMode::ReferenceHW,
                                                       numOfDPUGroups, numOfDMAPorts));

                buildReferenceHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions30XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for VPU30XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions30XX& options) {
                const auto numOfDPUGroups = convertToOptional(options.numberOfDPUGroups);
                const auto numOfDMAPorts = convertToOptional(options.numberOfDMAPorts);
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX30XX, VPU::CompilationMode::DefaultHW,
                                                       numOfDPUGroups, numOfDMAPorts));

                buildDefaultHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceSWOptions30XX>(
            "emu-reference-sw-mode",
            "Compile IE Network in EMU Reference Software mode (SW only execution) for VPU30XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options) {
                pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::VPUX30XX, VPU::CompilationMode::ReferenceSW));

                buildEMUReferenceSWModePipeline(pm, options);
            });
}

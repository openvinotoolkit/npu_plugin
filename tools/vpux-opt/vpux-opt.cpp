//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/IE/passes.hpp"
#include "vpux/compiler/VPU37XX/dialect/IE/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELF/passes.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/pipelines_register.hpp"
#include "vpux/compiler/tools/options.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        mlir::DialectRegistry registry;
        vpux::registerDialects(registry);
        // TODO: need to rework this unconditional replacement. Ticket: E#50937
        vpux::registerInterfacesWithReplacement(registry);

        const auto archKind = vpux::parseArchKind(argc, argv);

        const auto pipelineRegister = vpux::createPipelineRegister(archKind);
        pipelineRegister->registerPipelines();

        vpux::registerCorePasses();
        vpux::Const::registerConstPasses();
        vpux::EMU::registerEMUPasses();
        vpux::EMU::registerEMUPipelines();
        vpux::IE::registerIEPasses();
        vpux::IE::registerIEPipelines();
        vpux::IE::Arch30XX::registerIEPasses();
        vpux::IE::Arch37XX::registerIEPasses();
        vpux::VPU::registerVPUPasses();
        vpux::VPU::registerVPUPipelines();
        vpux::VPUIP::registerVPUIPPasses();
        vpux::VPUIP::registerVPUIPPipelines();
        vpux::VPURT::registerVPURTPipelines();
        vpux::VPURT::registerVPURTPasses();
        vpux::ELF::registerELFPasses();
        vpux::VPURegMapped::registerVPURegMappedPasses();
        vpux::VPUMI37XX::registerVPUMI37XXPasses();
        vpux::registerConversionPasses();
        vpux::registerConversionPipelines();

        mlir::registerTransformsPasses();
        mlir::func::registerFuncPasses();

        return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "VPUX Optimizer Testing Tool", registry, false));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

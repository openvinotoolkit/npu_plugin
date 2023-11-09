//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELF/passes.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/passes_register.hpp"
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
        // TODO: need to rework this unconditional replacement for dummy ops
        // there is an option for vpux-translate we can do it in the same way
        // Ticket: E#50937
        vpux::registerCommonInterfaces(registry, /*enableDummyOp*/ true);

        const auto hwSpecificRegistration = [&](vpux::StringRef helpHeader) {
            const auto archKind = vpux::parseArchKind(argc, argv, helpHeader);

            const auto pipelineRegistry = vpux::createPipelineRegistry(archKind);
            pipelineRegistry->registerPipelines();

            const auto passesRegistry = vpux::createPassesRegistry(archKind);
            passesRegistry->registerPasses();

            auto interfacesRegistry = vpux::createInterfacesRegistry(archKind);
            interfacesRegistry->registerInterfaces(registry);
        };

        vpux::registerCorePasses();
        vpux::Const::registerConstPasses();
        vpux::EMU::registerEMUPasses();
        vpux::EMU::registerEMUPipelines();
        vpux::IE::registerIEPasses();
        vpux::IE::registerIEPipelines();
        vpux::VPU::registerVPUPasses();
        vpux::VPU::registerVPUPipelines();
        vpux::VPUIP::registerVPUIPPasses();
        vpux::VPUIP::registerVPUIPPipelines();
        vpux::VPURT::registerVPURTPipelines();
        vpux::VPURT::registerVPURTPasses();
        vpux::ELF::registerELFPasses();
        vpux::VPUMI37XX::registerVPUMI37XXPasses();
        vpux::registerConversionPasses();
        vpux::registerConversionPipelines();

        mlir::registerTransformsPasses();
        mlir::func::registerFuncPasses();

        return mlir::asMainReturnCode(
                mlir::MlirOptMain(argc, argv, "NPU Optimizer Testing Tool", registry, false, hwSpecificRegistration));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

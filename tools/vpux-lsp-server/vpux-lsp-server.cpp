//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/pipelines.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include <mlir/Transforms/Passes.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        mlir::DialectRegistry registry;
        vpux::registerDialects(registry);

        vpux::registerCorePasses();
        vpux::Const::registerConstPasses();
        vpux::IE::registerIEPasses();
        vpux::IE::registerIEPipelines();
        vpux::VPUIP::registerVPUIPPasses();
        vpux::VPUIP::registerVPUIPPipelines();
        vpux::registerConversionPasses();
        vpux::registerConversionPipelines();
        vpux::VPU::registerVPUPipelines();
        vpux::registerPipelines();

        mlir::registerTransformsPasses();
        mlir::registerStandardPasses();

        return mlir::asMainReturnCode(mlir::MlirLspServerMain(argc, argv, registry));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/pipelines.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        mlir::DialectRegistry registry;
        vpux::registerDialects(registry);

        vpux::registerCorePasses();
        vpux::IE::registerIEPasses();
        vpux::IE::registerIEPipelines();
        vpux::IERT::registerIERTPasses();
        vpux::IERT::registerIERTPipelines();
        vpux::VPUIP::registerVPUIPPasses();
        vpux::registerConversionPasses();
        vpux::registerConversionPipelines();
        vpux::registerPipelines();

        mlir::registerTransformsPasses();
        mlir::registerStandardPasses();

        return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "VPUX Optimizer Testing Tool", registry, false));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

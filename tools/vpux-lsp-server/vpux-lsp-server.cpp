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

#ifdef ENABLE_PLAIDML
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/affinex/transforms/passes.h"
#include "pmlc/dialect/layer/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/transforms/passes.h"
#endif

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/SCF/Passes.h>
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
        vpux::IE::registerIEPasses();
        vpux::IE::registerIEPipelines();
        vpux::IERT::registerIERTPasses();
        vpux::IERT::registerIERTPipelines();
        vpux::VPUIP::registerVPUIPPasses();
        vpux::registerConversionPasses();
        vpux::registerConversionPipelines();
        vpux::registerPipelines();

        mlir::registerTransformsPasses();
        mlir::registerAffinePasses();
        mlir::registerSCFPasses();
        mlir::registerStandardPasses();

#ifdef ENABLE_PLAIDML
        pmlc::conversion::pxa_to_affine::registerPasses();
        pmlc::conversion::tile_to_pxa::registerPasses();
        pmlc::dialect::affinex::registerPasses();
        pmlc::dialect::layer::registerPasses();
        pmlc::dialect::pxa::registerPasses();
        pmlc::dialect::stdx::registerPasses();
        pmlc::dialect::tile::registerPasses();
        pmlc::transforms::registerPasses();
#endif

        return mlir::asMainReturnCode(mlir::MlirLspServerMain(argc, argv, registry));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

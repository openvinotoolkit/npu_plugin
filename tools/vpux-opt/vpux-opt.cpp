//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
#include <mlir/Transforms/Passes.h>

#include <cstdlib>
#include <iostream>

using namespace vpux;

int main(int argc, char* argv[]) {
    try {
        mlir::DialectRegistry registry;
        registerDialects(registry);

        registerCorePasses();
        IE::registerIEPasses();
        IE::registerPipelines();
        IERT::registerIERTPasses();
        VPUIP::registerVPUIPPasses();
        registerConversionPasses();
        registerConversionPipelines();
        registerPipelines();
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

        const auto res = mlir::MlirOptMain(argc, argv, "VPUX Optimizer Testing Tool", registry, false);
        return mlir::succeeded(res) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

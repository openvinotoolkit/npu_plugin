//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/conversion/passes.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/pipelines.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Support/MlirOptMain.h>

#include <iostream>

#include <cstdlib>

using namespace vpux;

int main(int argc, char* argv[]) {
    try {
        mlir::DialectRegistry registry;
        registry.insert<IE::IEDialect>();
        registry.insert<VPUIP::VPUIPDialect>();
        registry.insert<mlir::StandardOpsDialect>();

        IE::registerIEPasses();
        VPUIP::registerVPUIPPasses();
        registerConversionPasses();
        registerAllPipelines();

        const auto res = mlir::MlirOptMain(argc, argv, "VPUX Optimizer Testing Tool", registry, true);

        return mlir::succeeded(res) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

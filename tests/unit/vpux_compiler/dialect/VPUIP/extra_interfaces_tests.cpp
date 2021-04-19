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

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

TEST(MLIR_VPUIPExtraInterfaces, LayerExecutor) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) -> memref<1x512xf32> {
                %0 = memref.alloc() : memref<1x512xf32>
                %1 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x512xf32>) outputs(%0 : memref<1x512xf32>) -> memref<1x512xf32>
                %2 = IERT.Copy inputs(%1 : memref<1x512xf32>) outputs(%arg1 : memref<1x512xf32>) -> memref<1x512xf32>
                memref.dealloc %0 : memref<1x512xf32>
                return %2 : memref<1x512xf32>
            }
        }
    )";

    auto* iert = ctx.getOrLoadDialect<vpux::IERT::IERTDialect>();
    ASSERT_TRUE(iert != nullptr);

    const auto* info = iert->getRegisteredInterface<vpux::IERT::LayerInfoDialectInterface>();
    ASSERT_TRUE(info != nullptr);

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPUIP::createSetCompileParamsPass(vpux::VPUIP::ArchKind::VPU3700,
                                                       vpux::VPUIP::CompilationMode::ReferenceSW));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (mlir::isa<vpux::IERT::SoftMaxOp>(op)) {
            uint32_t numUnits = 0;
            auto kind = info->getExecutor(&op, numUnits);
            ASSERT_TRUE(kind != nullptr);
            ASSERT_TRUE(kind.isa<vpux::VPUIP::PhysicalProcessorAttr>());
            EXPECT_EQ(kind.cast<vpux::VPUIP::PhysicalProcessorAttr>().getValue(),
                      vpux::VPUIP::PhysicalProcessor::SHAVE_UPA);
            EXPECT_GE(numUnits, 1u);
        } else if (mlir::isa<vpux::IERT::CopyOp>(op)) {
            uint32_t numUnits = 0;
            auto kind = info->getExecutor(&op, numUnits);
            ASSERT_TRUE(kind != nullptr);
            ASSERT_TRUE(kind.isa<vpux::VPUIP::DMAEngineAttr>());
            EXPECT_EQ(kind.cast<vpux::VPUIP::DMAEngineAttr>().getValue(), vpux::VPUIP::DMAEngine::DMA_NN);
            EXPECT_EQ(numUnits, 1u);
        }
    }
}

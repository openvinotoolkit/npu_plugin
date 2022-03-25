//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

namespace {

void checkExecutorKind(mlir::Operation* op, vpux::VPU::ExecutorKind expectedKind) {
    auto iface = mlir::dyn_cast<vpux::IERT::AsyncLayerOpInterface>(op);
    ASSERT_NE(iface, nullptr);

    auto kindAttr = iface.getExecutor();
    ASSERT_TRUE(kindAttr != nullptr);
    ASSERT_TRUE(kindAttr.isa<mlir::SymbolRefAttr>());

    auto kind = vpux::VPU::symbolizeEnum<vpux::VPU::ExecutorKind>(kindAttr.getLeafName());
    EXPECT_EQ(kind.getValue(), expectedKind);
}

}  // namespace

TEST(MLIR_VPUIP_LayerInfo, AsyncLayerOpInterface) {
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

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::ReferenceSW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (mlir::isa<vpux::IERT::SoftMaxOp>(op)) {
            ::checkExecutorKind(&op, vpux::VPU::ExecutorKind::SHAVE_UPA);
        } else if (mlir::isa<vpux::IERT::CopyOp>(op)) {
            ::checkExecutorKind(&op, vpux::VPU::ExecutorKind::DMA_NN);
        }
    }
}

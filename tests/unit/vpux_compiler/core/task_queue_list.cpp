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

#include "vpux/compiler/core/task_queue_list.hpp"

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/Parser.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

TEST(MLIR_TaskQueueList, CheckTasksQueue) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: memref<1x100xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
                %0 = memref.alloc() : memref<1x100xf16>
                %1 = memref.alloc() : memref<1x100xf16>
                %token, %results = async.execute -> !async.value<memref<1x100xf16>> attributes {IERT.executor = @SHAVE_UPA, IERT.num_units = 16 : i64, "async-deps-index" = 0 : i64} {
                    %3 = IERT.ReLU inputs(%arg0 : memref<1x100xf16>) outputs(%0 : memref<1x100xf16>) -> memref<1x100xf16>
                    async.yield %3 : memref<1x100xf16>
                }
                %token_0, %results_1 = async.execute [%token] (%results as %arg2: !async.value<memref<1x100xf16>>) -> !async.value<memref<1x100xf16>> attributes {IERT.executor = @SHAVE_UPA, IERT.num_units = 16 : i64, "async-deps-index" = 1 : i64} {
                    %3 = IERT.ReLU inputs(%arg2 : memref<1x100xf16>) outputs(%1 : memref<1x100xf16>) -> memref<1x100xf16>
                    async.yield %3 : memref<1x100xf16>
                }
                %token_2, %results_3 = async.execute [%token_0] (%results_1 as %arg2: !async.value<memref<1x100xf16>>) -> !async.value<memref<100xf16>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
                    %3 = IERT.GenericReshape inputs(%arg2 : memref<1x100xf16>) -> memref<100xf16>
                    %4 = IERT.Copy inputs(%3 : memref<100xf16>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>
                    async.yield %4 : memref<100xf16>
                }
                %2 = async.await %results_3 : !async.value<memref<100xf16>>
                return %2 : memref<100xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::TaskQueueList taskQueueList{func};

    ASSERT_TRUE(taskQueueList.size() == 2);

    auto shaveExecAttr = vpux::IndexedSymbolAttr::get(&ctx, "SHAVE_UPA");
    auto dmaExecAttr = vpux::IndexedSymbolAttr::get(&ctx, "DMA_NN");
    auto nceExecAttr = vpux::IndexedSymbolAttr::get(&ctx, "NCE");

    ASSERT_TRUE(taskQueueList.hasQueue(shaveExecAttr));
    ASSERT_TRUE(taskQueueList.hasQueue(dmaExecAttr));
    ASSERT_TRUE(!taskQueueList.hasQueue(nceExecAttr));

    auto shaveTasks = taskQueueList.getQueue(shaveExecAttr);
    auto dmaTasks = taskQueueList.getQueue(dmaExecAttr);

    ASSERT_TRUE(shaveTasks.size() == 2);
    ASSERT_TRUE(dmaTasks.size() == 1);
}

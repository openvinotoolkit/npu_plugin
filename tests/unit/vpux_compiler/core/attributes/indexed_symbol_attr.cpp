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

#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

constexpr vpux::StringRef NCE_NAME = "NCE";
constexpr vpux::StringRef DPU_NAME = "DPU";

void checkDDRSpace(vpux::IndexedSymbolAttr indexedSymbol) {
    ASSERT_EQ(indexedSymbol.getRootName(), DDR_NAME);
    ASSERT_EQ(indexedSymbol.getLeafName(), DDR_NAME);
    ASSERT_FALSE(indexedSymbol.getIndex().hasValue());
    ASSERT_FALSE(indexedSymbol.getNestedReference().hasValue());
}

void checkCMXSpace(vpux::IndexedSymbolAttr indexedSymbol, int64_t expIndex) {
    ASSERT_EQ(indexedSymbol.getRootName(), CMX_NAME);
    ASSERT_EQ(indexedSymbol.getLeafName(), CMX_NAME);
    ASSERT_TRUE(indexedSymbol.getIndex().hasValue());
    ASSERT_EQ(indexedSymbol.getIndex().getValue(), expIndex);
    ASSERT_FALSE(indexedSymbol.getNestedReference().hasValue());
}

}  // namespace

TEST(MLIR_IndexedSymbolAttr, CheckNestedAttr) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    const int64_t nestedIdx = 0;
    const auto nestedNameAttr = mlir::FlatSymbolRefAttr::get(&ctx, "@DUMMY");
    const auto nestedAttr = vpux::IndexedSymbolAttr::get(&ctx, {nestedNameAttr, vpux::getIntAttr(&ctx, nestedIdx)});

    const int64_t rootIdx = 1;
    const auto rootNameAttr = mlir::FlatSymbolRefAttr::get(&ctx, CMX_NAME);
    const auto rootAttr = vpux::IndexedSymbolAttr::get(&ctx, {rootNameAttr, vpux::getIntAttr(&ctx, rootIdx), nestedAttr});

    const auto fullSymbolAttr = mlir::SymbolRefAttr::get(rootNameAttr.getAttr(), llvm::makeArrayRef(nestedNameAttr));

    ASSERT_EQ(rootAttr.getRootReference(), rootNameAttr);
    ASSERT_EQ(rootAttr.getLeafReference(), nestedNameAttr);
    ASSERT_EQ(rootAttr.getFullReference(), fullSymbolAttr);
    ASSERT_TRUE(rootAttr.getIndex().hasValue());
    ASSERT_EQ(rootAttr.getIndex().getValue(), rootIdx);
    ASSERT_TRUE(rootAttr.getNestedReference().hasValue());

    const auto actNestedAttr = rootAttr.getNestedReference().getValue();
    ASSERT_EQ(actNestedAttr, nestedAttr);
    ASSERT_EQ(actNestedAttr.getRootReference(), nestedNameAttr);
    ASSERT_EQ(actNestedAttr.getLeafReference(), nestedNameAttr);
    ASSERT_EQ(actNestedAttr.getFullReference(), nestedNameAttr);
    ASSERT_TRUE(actNestedAttr.getIndex().hasValue());
    ASSERT_EQ(actNestedAttr.getIndex().getValue(), nestedIdx);
    ASSERT_FALSE(actNestedAttr.getNestedReference().hasValue());
}

TEST(MLIR_IndexedSymbolAttr, CheckMemoryResourceAttr) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: memref<1x8x20x20xf16, @DDR>, %arg1: memref<1x8x20x20xf16, @DDR>) -> memref<1x8x20x20xf16, @DDR> {
                %0 = memref.alloc(): memref<1x8x20x20xf16, [@CMX_NN, 0]>
                %1 = IERT.Copy inputs(%arg0 : memref<1x8x20x20xf16, @DDR>) outputs(%0 : memref<1x8x20x20xf16, [@CMX_NN, 0]>) -> memref<1x8x20x20xf16, [@CMX_NN, 0]>
                %2 = IERT.Copy inputs(%0 : memref<1x8x20x20xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x8x20x20xf16, @DDR>) -> memref<1x8x20x20xf16, @DDR>

                return %2 : memref<1x8x20x20xf16, @DDR>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto checkMemSpace = [](vpux::IndexedSymbolAttr indexedSymAttr) {
        if (indexedSymAttr.getLeafName() == DDR_NAME) {
            checkDDRSpace(indexedSymAttr);
        } else {
            checkCMXSpace(indexedSymAttr, 0);
        }
    };

    for (auto& op : func.getOps()) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto type = allocOp.memref().getType().cast<mlir::MemRefType>();
            auto memSpace = vpux::getMemorySpace(type);

            checkCMXSpace(memSpace, 0);
        } else if (auto copyOp = mlir::dyn_cast<vpux::IERT::CopyOp>(op)) {
            auto inMemSpace = vpux::getMemorySpace(copyOp.input().getType().cast<mlir::MemRefType>());
            auto outMemSpace = vpux::getMemorySpace(copyOp.output().getType().cast<mlir::MemRefType>());

            ASSERT_NE(inMemSpace, outMemSpace);
            ASSERT_TRUE(inMemSpace.getLeafName() == DDR_NAME || inMemSpace.getLeafName() == CMX_NAME);

            checkMemSpace(inMemSpace);
            checkMemSpace(outMemSpace);
        }
    }
}

TEST(MLIR_IndexedSymbolAttr, CheckExecutorResourceAttr) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            IE.ExecutorResource 16 of @SHAVE_UPA
            IE.ExecutorResource 4 of  @NCE {
                IE.ExecutorResource 5 of @DPU
            }
            IE.ExecutorResource 1 of @DMA_NN

            func @main(%arg0: memref<1x16x62x62xf16, #NHWC>,
                        %arg1: memref<1x48x60x60xf16, #NHWC>) -> memref<1x48x60x60xf16, #NHWC> {
                %cst = const.Declare memref<48x16x3x3xf16, #NHWC> = #const.Content<dense<1.000000e+00> : tensor<48x16x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]>
                %0 = IERT.StaticAlloc<0> -> memref<1x16x62x62xf16, #NHWC, @CMX_NN>
                %1 = IERT.StaticAlloc<468608> -> memref<48x16x3x3xf16, #NHWC, @CMX_NN>
                %2 = IERT.StaticAlloc<123008> -> memref<1x48x60x60xf16, #NHWC, @CMX_NN>
                %3 = IERT.StaticAlloc<482432> -> memref<48x1x1x4xsi32, @CMX_NN>
                %token, %results = async.execute ->
                                    !async.value<memref<1x16x62x62xf16, #NHWC, @CMX_NN>>
                                        attributes { IERT.executor = @DMA_NN } {
                    %5 = IERT.Copy inputs(%arg0 : memref<1x16x62x62xf16, #NHWC>)
                                   outputs(%0 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>) -> memref<1x16x62x62xf16, #NHWC, @CMX_NN>
                    async.yield %0 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>
                }
                %token_0, %results_1:2 = async.execute [%token] ->
                                            (!async.value<memref<48x1x1x4xsi32, @CMX_NN>>, !async.value<memref<48x16x3x3xf16, #NHWC, @CMX_NN>>)
                                                attributes { IERT.executor = @DMA_NN } {
                  %cst_6 = const.Declare memref<48x1x1x4xsi32> = #const.Content<dense<1> : tensor<48x1x1x4xsi32>>
                  %5 = IERT.Copy inputs(%cst_6 : memref<48x1x1x4xsi32>) outputs(%3 : memref<48x1x1x4xsi32, @CMX_NN>) -> memref<48x1x1x4xsi32, @CMX_NN>
                  %6 = IERT.Copy inputs(%cst : memref<48x16x3x3xf16, #NHWC>) outputs(%1 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>) -> memref<48x16x3x3xf16, #NHWC, @CMX_NN>
                  async.yield %3, %1 : memref<48x1x1x4xsi32, @CMX_NN>, memref<48x16x3x3xf16, #NHWC, @CMX_NN>
                }
                %token_2, %results_3 = async.execute [%token_0] (
                                        %results as %arg2: !async.value<memref<1x16x62x62xf16, #NHWC, @CMX_NN>>,
                                        %results_1#1 as %arg3: !async.value<memref<48x16x3x3xf16, #NHWC, @CMX_NN>>,
                                        %results_1#0 as %arg4: !async.value<memref<48x1x1x4xsi32, @CMX_NN>>) ->
                                            !async.value<memref<1x48x60x60xf16, #NHWC, @CMX_NN>>
                                                attributes { IERT.executor = [@NCE, 1, [@DPU]] } {
                  %5 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
                                                input(%arg2 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
                                                weights(%arg3 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
                                                weight_table(%arg4 : memref<48x1x1x4xsi32, @CMX_NN>)
                                                parent_input(%arg2 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
                                                parent_output(%2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
                                                outputs(%2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) ->
                memref<1x48x60x60xf16, #NHWC, @CMX_NN> variants :  {
                    DPUTask {end = [59, 11, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
                    DPUTask {end = [59, 23, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
                    DPUTask {end = [59, 35, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 24, 0]}
                    DPUTask {end = [59, 47, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 36, 0]}
                    DPUTask {end = [59, 59, 47], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 48, 0]}
                  } PPE :  {
                  }
                  async.yield %2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>
                }
                %token_4, %results_5 = async.execute [%token_2] (%results_3 as %arg2: !async.value<memref<1x48x60x60xf16, #NHWC, @CMX_NN>>) ->
                                        !async.value<memref<1x48x60x60xf16, #NHWC>>
                                            attributes { IERT.executor = @DMA_NN } {
                  %5 = IERT.Copy inputs(%arg2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x48x60x60xf16, #NHWC>) -> memref<1x48x60x60xf16, #NHWC>
                  async.yield %arg1 : memref<1x48x60x60xf16, #NHWC>
                }
                %4 = async.await %results_5 : !async.value<memref<1x48x60x60xf16, #NHWC>>
                return %4 : memref<1x48x60x60xf16, #NHWC>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    for (auto& op : func.getOps()) {
        if(auto executeOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            const auto executor = vpux::IERT::IERTDialect::getExecutor(executeOp);
            ASSERT_TRUE(executor != nullptr);

            auto execRes = vpux::IE::getAvailableExecutor(module.get(), executor.getRootReference());
            ASSERT_TRUE(execRes != nullptr);

            if (executor.getRootName() == NCE_NAME) {
                ASSERT_TRUE(executor.getIndex().hasValue());
                ASSERT_EQ(executor.getIndex().getValue(), 1);
                ASSERT_TRUE(executor.getNestedReference().hasValue());
                ASSERT_EQ(executor.getLeafName(), DPU_NAME);

                auto nestedExecAttr = executor.getNestedReference().getValue();
                ASSERT_EQ(nestedExecAttr.getRootName(), DPU_NAME);
                ASSERT_FALSE(nestedExecAttr.getNestedReference().hasValue());

                auto nestedExecRes = vpux::IE::getAvailableExecutor(module.get(), executor.getFullReference());
                ASSERT_TRUE(nestedExecRes != nullptr);
                ASSERT_EQ(nestedExecRes.sym_name(), DPU_NAME);
            }
        }
    }
}

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

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/Parser.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

void checkDDRSpace(vpux::IndexedSymbolAttr indexedSymbol) {
    ASSERT_TRUE(indexedSymbol.getName() == DDR_NAME);
    ASSERT_TRUE(!indexedSymbol.isDefined());
    ASSERT_TRUE(!indexedSymbol.getNestedAttr().hasValue());
}

void checkCMXSpace(vpux::IndexedSymbolAttr indexedSymbol, int64_t expIndex) {
    ASSERT_TRUE(indexedSymbol.getName() == CMX_NAME);
    ASSERT_TRUE(indexedSymbol.isDefined());
    ASSERT_TRUE(indexedSymbol.getIndex() == expIndex);
    ASSERT_TRUE(!indexedSymbol.getNestedAttr().hasValue());
}

}

TEST(MLIR_IndexedSymbolAttr, CheckNestedAttr) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    int64_t dummyIdx = 0;
    auto dummyNameAttr = mlir::FlatSymbolRefAttr::get(&ctx, "@DUMMY");
    auto expNestedAttr = vpux::IndexedSymbolAttr::get(&ctx, {dummyNameAttr, vpux::getIntAttr(&ctx, dummyIdx)});

    int64_t cmxIdx = 1;
    auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(&ctx, CMX_NAME);
    auto rootAttr = vpux::IndexedSymbolAttr::get(&ctx, {cmxNameAttr, vpux::getIntAttr(&ctx, cmxIdx), expNestedAttr});

    ASSERT_TRUE(rootAttr.getName() == CMX_NAME);
    ASSERT_TRUE(rootAttr.getIndex() == cmxIdx);
    ASSERT_TRUE(rootAttr.getNestedAttr().hasValue());

    auto actNestedAttr = rootAttr.getNestedAttr().getValue();
    ASSERT_TRUE(actNestedAttr.getNameAttr() == dummyNameAttr);
    ASSERT_TRUE(actNestedAttr.getIndex() == dummyIdx);
    ASSERT_TRUE(!actNestedAttr.getNestedAttr().hasValue());
}

TEST(MLIR_IndexedSymbolAttr, CheckParsedAttr) {
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
        if(indexedSymAttr.getName() == DDR_NAME) {
            checkDDRSpace(indexedSymAttr);
        } else {
            checkCMXSpace(indexedSymAttr, 0);
        }
    };

    for (auto& op : func.getOps()) {
        if(auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto type = allocOp.memref().getType().cast<mlir::MemRefType>();
            auto memSpace = vpux::getMemorySpace(type);

            checkCMXSpace(memSpace, 0);
        } else if (auto copyOp = mlir::dyn_cast<vpux::IERT::CopyOp>(op)) {
            auto inMemSpace = vpux::getMemorySpace(copyOp.input().getType().cast<mlir::MemRefType>());
            auto outMemSpace = vpux::getMemorySpace(copyOp.output().getType().cast<mlir::MemRefType>());

            ASSERT_TRUE(inMemSpace != outMemSpace);
            ASSERT_TRUE(inMemSpace.getName() == DDR_NAME || inMemSpace.getName() == CMX_NAME);

            checkMemSpace(inMemSpace);
            checkMemSpace(outMemSpace);
        }
    }
}

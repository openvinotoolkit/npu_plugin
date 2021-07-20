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

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

namespace {

class TestMultiViewOp : public mlir::Op<TestMultiViewOp, vpux::MultiViewOpInterface::Trait> {
public:
    using Op::Op;

    static constexpr llvm::StringLiteral getOperationName() {
        return llvm::StringLiteral("test.multiview");
    }

    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
        return {};
    }

    mlir::Value getViewSource(ptrdiff_t resIndex) {
        return (*this)->getOperand((*this)->getNumOperands() - (*this)->getNumResults() + resIndex);
    }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::TypeRange resultTypes,
                      mlir::ValueRange operands, llvm::ArrayRef<mlir::NamedAttribute> attributes = {}) {
        state.addTypes(resultTypes);
        state.addOperands(operands);
        state.addAttributes(attributes);
    }
};

class TestDialect final : public mlir::Dialect {
public:
    explicit TestDialect(mlir::MLIRContext* ctx)
            : mlir::Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<TestDialect>()) {
        addOperations<TestMultiViewOp>();
    }

    static constexpr llvm::StringLiteral getDialectNamespace() {
        return llvm::StringLiteral("test");
    }
};

}  // namespace

TEST(MLIR_AliasesInfo, ValidCases) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %0 = memref.alloc(): memref<100xf32>
                %1 = memref.subview %arg[0][50][1] : memref<100xf32> to memref<50xf32>
                %2:2 = "test.multiview"(%0, %1) : (memref<100xf32>, memref<50xf32>) -> (memref<100xf32>, memref<50xf32>)
                return %2#0 : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto funcArgRoot = info.getRoot(funcArg);
    EXPECT_TRUE(funcArgRoot == funcArg);

    const auto& funcArgAliases = info.getAliases(funcArg);
    EXPECT_EQ(funcArgAliases.size(), 3) << "%arg aliases: %arg, %1, %2#1";

    for (const auto alias : funcArgAliases) {
        if (auto* producerOp = alias.getDefiningOp()) {
            EXPECT_TRUE(mlir::isa<mlir::memref::SubViewOp>(producerOp) || mlir::isa<TestMultiViewOp>(producerOp));

            if (mlir::isa<TestMultiViewOp>(producerOp)) {
                EXPECT_EQ(alias.cast<mlir::OpResult>().getResultNumber(), 1);
            }
        } else {
            EXPECT_TRUE(alias == funcArg);
        }
    }

    func.walk([&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto allocRes = allocOp.getResult();

            const auto allocRoot = info.getRoot(allocRes);
            EXPECT_TRUE(allocRoot == allocRes);

            const auto& allocAliases = info.getAliases(allocRes);
            EXPECT_EQ(allocAliases.size(), 2) << "%0 aliases: %0, %2#0";
            for (const auto alias : allocAliases) {
                auto* producerOp = alias.getDefiningOp();
                ASSERT_TRUE(producerOp != nullptr);

                EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp) || mlir::isa<TestMultiViewOp>(producerOp))
                    << "producerOp = " << producerOp->getName().getStringRef().data();

                if (mlir::isa<TestMultiViewOp>(producerOp)) {
                    EXPECT_EQ(alias.cast<mlir::OpResult>().getResultNumber(), 0);
                } else {
                    EXPECT_TRUE(producerOp == allocOp);
                }
            }
        } else if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
            const auto viewRes = viewOp.getResult();

            const auto viewRoot = info.getRoot(viewRes);
            EXPECT_TRUE(viewRoot.isa<mlir::BlockArgument>());
        } else if (auto viewOp = mlir::dyn_cast<TestMultiViewOp>(op)) {
            const auto viewRes0 = viewOp->getResult(0);
            const auto viewRoot0 = info.getRoot(viewRes0);
            EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(viewRoot0.getDefiningOp()));

            const auto viewRes1 = viewOp->getResult(1);
            const auto viewRoot1 = info.getRoot(viewRes1);
            EXPECT_TRUE(viewRoot1.isa<mlir::BlockArgument>());
        }
    });
}

TEST(MLIR_AliasesInfo, AsyncRegions) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::async::AsyncDialect>();
    registry.insert<mlir::StandardOpsDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg: memref<100xf32>) -> memref<70xf32> {
                %0 = memref.subview %arg[0][90][1] : memref<100xf32> to memref<90xf32>

                %t1, %f1 =
                    async.execute () -> !async.value<memref<80xf32>>
                    {
                        %1 = memref.subview %0[0][80][1] : memref<90xf32> to memref<80xf32>
                        async.yield %1 : memref<80xf32>
                    }

                %t2, %f2 =
                    async.execute [%t1](%f1 as %1 : !async.value<memref<80xf32>>) -> !async.value<memref<70xf32>>
                    {
                        %2 = memref.subview %1[0][70][1] : memref<80xf32> to memref<70xf32>
                        async.yield %2 : memref<70xf32>
                    }

                %2 = async.await %f2 : !async.value<memref<70xf32>>

                return %2 : memref<70xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto& aliases = info.getAliases(funcArg);
    EXPECT_EQ(aliases.size(), 8) << "%arg aliases: %arg, %0, %1+%f1, %1+%2+%f2, %2";
}

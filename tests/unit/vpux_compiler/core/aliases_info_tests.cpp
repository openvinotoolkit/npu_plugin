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

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"

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

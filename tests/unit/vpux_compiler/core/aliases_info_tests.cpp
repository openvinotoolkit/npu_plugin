//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace AlisesInfoTest {

class TestMultiViewOp : public mlir::Op<TestMultiViewOp, vpux::MultiViewOpInterface::Trait> {
public:
    using Op::Op;

    static constexpr StringLiteral getOperationName() {
        return StringLiteral("test.multiview");
    }

    static ArrayRef<StringRef> getAttributeNames() {
        return {};
    }

    mlir::Value getViewSource(ptrdiff_t resIndex) {
        return (*this)->getOperand((*this)->getNumOperands() - (*this)->getNumResults() + resIndex);
    }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::TypeRange resultTypes,
                      mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes = {}) {
        state.addTypes(resultTypes);
        state.addOperands(operands);
        state.addAttributes(attributes);
    }
};

class TestGroupedViewOp : public mlir::Op<TestGroupedViewOp, vpux::GroupedViewOpInterface::Trait> {
public:
    using Op::Op;

    static constexpr StringLiteral getOperationName() {
        return StringLiteral("test.groupedview");
    }

    static ArrayRef<StringRef> getAttributeNames() {
        return {};
    }

    mlir::ValueRange getViewSources() {
        return (*this)->getOperands();
    }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::TypeRange resultTypes,
                      mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes = {}) {
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
        addOperations<TestGroupedViewOp>();
    }

    static constexpr StringLiteral getDialectNamespace() {
        return StringLiteral("test");
    }
};

}  // namespace AlisesInfoTest

TEST(MLIR_AliasesInfo, TestMultiViewOp) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %0 = memref.alloc(): memref<100xf32>
                %1 = memref.subview %arg[0][50][1] : memref<100xf32> to memref<50xf32>
                %2:2 = "test.multiview"(%0, %1) : (memref<100xf32>, memref<50xf32>) -> (memref<100xf32>, memref<50xf32>)
                return %2#0 : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto funcArgSource = info.getSource(funcArg);
    EXPECT_TRUE(funcArgSource == nullptr);

    const auto funcArgRoots = info.getRoots(funcArg);
    EXPECT_EQ(funcArgRoots.size(), 1) << "funcArg roots: %arg";
    EXPECT_TRUE(*funcArgRoots.begin() == funcArg);

    const auto& funcArgAliases = info.getAllAliases(funcArg);
    EXPECT_EQ(funcArgAliases.size(), 3) << "%arg aliases: %arg, %1, %2#1";

    for (const auto alias : funcArgAliases) {
        if (auto* producerOp = alias.getDefiningOp()) {
            EXPECT_TRUE(mlir::isa<mlir::memref::SubViewOp>(producerOp) ||
                        mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp));

            if (mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp)) {
                EXPECT_EQ(alias.cast<mlir::OpResult>().getResultNumber(), 1);
            }
        } else {
            EXPECT_TRUE(alias == funcArg);
        }
    }

    func.walk([&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto allocRes = allocOp.getResult();

            const auto allocSource = info.getSource(allocRes);
            EXPECT_TRUE(allocSource == nullptr);

            const auto allocRoots = info.getRoots(allocRes);
            EXPECT_EQ(allocRoots.size(), 1) << "allocRes roots: %0";
            EXPECT_TRUE(*allocRoots.begin() == allocRes);

            const auto& allocAliases = info.getAllAliases(allocRes);
            EXPECT_EQ(allocAliases.size(), 2) << "%0 aliases: %0, %2#0";
            for (const auto alias : allocAliases) {
                auto* producerOp = alias.getDefiningOp();
                ASSERT_TRUE(producerOp != nullptr);

                EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp) ||
                            mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp))
                        << "producerOp = " << producerOp->getName().getStringRef().data();

                if (mlir::isa<AlisesInfoTest::TestMultiViewOp>(producerOp)) {
                    EXPECT_EQ(alias.cast<mlir::OpResult>().getResultNumber(), 0);
                } else {
                    EXPECT_TRUE(producerOp == allocOp);
                }
            }
        } else if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
            const auto viewRes = viewOp.getResult();

            const auto viewSource = info.getSource(viewRes);
            EXPECT_TRUE(viewSource == viewOp.getViewSource());

            const auto viewRoots = info.getRoots(viewRes);
            EXPECT_EQ(viewRoots.size(), 1) << "viewRes roots: %arg";
            EXPECT_TRUE((*viewRoots.begin()).isa<mlir::BlockArgument>());
        } else if (auto viewOp = mlir::dyn_cast<AlisesInfoTest::TestMultiViewOp>(op)) {
            const auto viewRes0 = viewOp->getResult(0);

            const auto viewSource0 = info.getSource(viewRes0);
            EXPECT_TRUE(viewSource0 == viewOp.getViewSource(0));

            const auto viewRoots0 = info.getRoots(viewRes0);
            EXPECT_EQ(viewRoots0.size(), 1) << "viewRes0 roots: %0";
            EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>((*viewRoots0.begin()).getDefiningOp()));

            const auto viewRes1 = viewOp->getResult(1);

            const auto viewSource1 = info.getSource(viewRes1);
            EXPECT_TRUE(viewSource1 == viewOp.getViewSource(1));

            const auto viewRoots1 = info.getRoots(viewRes1);
            EXPECT_EQ(viewRoots1.size(), 1) << "viewRes1 roots: %arg";
            EXPECT_TRUE((*viewRoots1.begin()).isa<mlir::BlockArgument>());
        }
    });
}

TEST(MLIR_AliasesInfo, TestGroupedViewOp) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<AlisesInfoTest::TestDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<100xf32> {
                %0 = memref.alloc(): memref<50xf32>
                %1 = "test.groupedview"(%arg, %0) : (memref<100xf32>, memref<50xf32>) -> memref<100xf32>
                return %1 : memref<100xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto funcArgSource = info.getSource(funcArg);
    EXPECT_TRUE(funcArgSource == nullptr);

    const auto funcArgRoots = info.getRoots(funcArg);
    EXPECT_EQ(funcArgRoots.size(), 1) << "funcArg roots: %arg";
    EXPECT_TRUE(*funcArgRoots.begin() == funcArg);

    const auto& funcArgAliases = info.getAllAliases(funcArg);
    EXPECT_EQ(funcArgAliases.size(), 2) << "%arg aliases: %arg, %1";

    for (const auto alias : funcArgAliases) {
        if (auto* producerOp = alias.getDefiningOp()) {
            EXPECT_TRUE(mlir::isa<AlisesInfoTest::TestGroupedViewOp>(producerOp));
        } else {
            EXPECT_TRUE(alias == funcArg);
        }
    }

    func.walk([&](mlir::Operation* op) {
        if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            const auto allocRes = allocOp.getResult();

            const auto allocSource = info.getSource(allocRes);
            EXPECT_TRUE(allocSource == nullptr);

            const auto allocRoots = info.getRoots(allocRes);
            EXPECT_EQ(allocRoots.size(), 1) << "allocRes roots: %0";
            EXPECT_TRUE(*allocRoots.begin() == allocRes);

            const auto& allocAliases = info.getAllAliases(allocRes);
            EXPECT_EQ(allocAliases.size(), 2) << "%0 aliases: %0, %1";
            for (const auto alias : allocAliases) {
                auto* producerOp = alias.getDefiningOp();
                ASSERT_TRUE(producerOp != nullptr);

                EXPECT_TRUE(mlir::isa<mlir::memref::AllocOp>(producerOp) ||
                            mlir::isa<AlisesInfoTest::TestGroupedViewOp>(producerOp))
                        << "producerOp = " << producerOp->getName().getStringRef().data();
            }
        } else if (auto viewOp = mlir::dyn_cast<AlisesInfoTest::TestGroupedViewOp>(op)) {
            const auto viewRes = viewOp->getResult(0);

            const auto viewSources = info.getSources(viewRes);
            EXPECT_EQ(viewSources.size(), 2) << "test.groupedview sources: %arg, %0";
            EXPECT_TRUE(viewSources.size() == viewOp.getViewSources().size());
            for (const auto& source : viewOp.getViewSources()) {
                EXPECT_TRUE(viewSources.count(source) > 0);
            }

            const auto viewRoots = info.getRoots(viewRes);
            EXPECT_EQ(viewRoots.size(), 2) << "test.groupedview roots: %arg, %0";
            for (const auto& root : viewRoots) {
                EXPECT_TRUE(root.isa<mlir::BlockArgument>() || mlir::isa<mlir::memref::AllocOp>(root.getDefiningOp()));
            }
        }
    });
}

TEST(MLIR_AliasesInfo, AsyncRegions) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::async::AsyncDialect>();
    registry.insert<mlir::func::FuncDialect>();

    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg: memref<100xf32>) -> memref<70xf32> {
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

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfo info(func);

    const auto funcArg = func.getArgument(0);

    const auto& aliases = info.getAllAliases(funcArg);
    EXPECT_EQ(aliases.size(), 8) << "%arg aliases: %arg, %0, %1+%f1, %1+%2+%f2, %2";
}

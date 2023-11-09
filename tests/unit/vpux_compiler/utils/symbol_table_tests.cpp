//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

namespace mlir {
/// Generates a new symbol reference attribute with a new leaf reference.
static SymbolRefAttr generateNewRefAttr(SymbolRefAttr oldAttr, SymbolRefAttr newLeafAttr) {
    if (oldAttr.isa<FlatSymbolRefAttr>()) {
        return newLeafAttr;
    }
    auto nestedRefs = llvm::to_vector<2>(oldAttr.getNestedReferences());
    nestedRefs.back() = FlatSymbolRefAttr::get(newLeafAttr.getRootReference());

    nestedRefs.append(newLeafAttr.getNestedReferences().begin(), newLeafAttr.getNestedReferences().end());

    return SymbolRefAttr::get(oldAttr.getRootReference(), nestedRefs);
}
}  // namespace mlir

using MLIR_SymbolTable = MLIR_UnitBase;

TEST_F(MLIR_SymbolTable, CheckGenerateNewRefAttr) {
    mlir::MLIRContext ctx(registry);

    llvm::StringLiteral root = "root";
    llvm::StringLiteral alpha = "alpha";
    llvm::StringLiteral beta = "beta";
    llvm::StringLiteral gamma = "gamma";
    llvm::StringLiteral theta = "theta";
    mlir::SymbolRefAttr test;

    auto rootAttr = mlir::StringAttr::get(&ctx, root);
    auto alphaAttr = mlir::FlatSymbolRefAttr::get(&ctx, alpha);
    auto betaAttr = mlir::FlatSymbolRefAttr::get(&ctx, beta);
    auto gammaAttr = mlir::FlatSymbolRefAttr::get(&ctx, gamma);
    auto thetaAttr = mlir::FlatSymbolRefAttr::get(&ctx, theta);

    test = mlir::generateNewRefAttr(alphaAttr, betaAttr);
    ASSERT_EQ(test, betaAttr);

    auto ralpha = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr});
    auto rbeta = mlir::SymbolRefAttr::get(rootAttr, {betaAttr});
    test = mlir::generateNewRefAttr(ralpha, betaAttr);
    ASSERT_EQ(test, rbeta);

    auto rabg = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, betaAttr, gammaAttr});
    auto rabt = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, betaAttr, thetaAttr});
    test = mlir::generateNewRefAttr(rabg, thetaAttr);
    ASSERT_EQ(test, rabt);

    auto rab = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, betaAttr});
    auto gt = mlir::SymbolRefAttr::get(gammaAttr.getAttr(), {thetaAttr});
    auto ragt = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, gammaAttr, thetaAttr});
    test = mlir::generateNewRefAttr(rab, gt);
    ASSERT_EQ(test, ragt);
}

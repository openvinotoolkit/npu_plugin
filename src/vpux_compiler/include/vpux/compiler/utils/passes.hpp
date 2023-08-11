//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace vpux {

//
// Options
//

using IntOption = mlir::detail::PassOptions::Option<int>;
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;
using DoubleOption = mlir::detail::PassOptions::Option<double>;

Optional<int> convertToOptional(const IntOption& intOption);
Optional<std::string> convertToOptional(const StrOption& strOption);
bool isOptionEnabled(const BoolOption& option);

//
// PatternBenefit
//

extern const mlir::PatternBenefit benefitLow;
extern const mlir::PatternBenefit benefitMid;
extern const mlir::PatternBenefit benefitHigh;

SmallVector<mlir::PatternBenefit> getBenefitLevels(uint32_t levels);

//
// FunctionPass
//

class FunctionPass : public mlir::OperationPass<mlir::func::FuncOp> {
public:
    using mlir::OperationPass<mlir::func::FuncOp>::OperationPass;

protected:
    void initLogger(Logger log, StringLiteral passName);

protected:
    virtual void safeRunOnFunc() = 0;

protected:
    Logger _log = Logger::global();

private:
    void runOnOperation() final;
};

//
// ModulePass
//

class ModulePass : public mlir::OperationPass<mlir::ModuleOp> {
public:
    using mlir::OperationPass<mlir::ModuleOp>::OperationPass;

protected:
    void initLogger(Logger log, StringLiteral passName);

protected:
    virtual void safeRunOnModule() = 0;

protected:
    Logger _log = Logger::global();

private:
    void runOnOperation() final;
};

}  // namespace vpux

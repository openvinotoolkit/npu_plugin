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

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace vpux {

//
// PatternBenefit
//

extern const mlir::PatternBenefit benefitLow;
extern const mlir::PatternBenefit benefitMid;
extern const mlir::PatternBenefit benefitHigh;

//
// FunctionPass
//

class FunctionPass : public mlir::FunctionPass {
protected:
    using mlir::FunctionPass::FunctionPass;

protected:
    void initLogger(Logger log, StringLiteral passName);

protected:
    virtual void safeRunOnFunc() = 0;

protected:
    Logger _log = Logger::global();

private:
    void runOnFunction() final;
};

//
// ModulePass
//

class ModulePass : public mlir::OperationPass<mlir::ModuleOp> {
protected:
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

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// PatternBenefit
//

const mlir::PatternBenefit vpux::benefitLow(1);
const mlir::PatternBenefit vpux::benefitMid(2);
const mlir::PatternBenefit vpux::benefitHigh(3);

//
// FunctionPass
//

void vpux::FunctionPass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::FunctionPass::runOnFunction() {
    try {
        _log.trace("Run on Function '{0}'", getFunction().getName());

        _log = _log.nest();
        safeRunOnFunc();
        _log = _log.unnest();
    } catch (const std::exception& e) {
        (void)errorAt(getFunction(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// ModulePass
//

void vpux::ModulePass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::ModulePass::runOnOperation() {
    try {
        safeRunOnModule();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

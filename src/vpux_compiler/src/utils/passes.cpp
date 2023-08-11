//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Options
//

Optional<int> vpux::convertToOptional(const IntOption& intOption) {
    if (intOption.hasValue()) {
        return intOption.getValue();
    }
    return None;
}

Optional<std::string> vpux::convertToOptional(const StrOption& strOption) {
    if (strOption.hasValue()) {
        return strOption.getValue();
    }
    return None;
}

bool vpux::isOptionEnabled(const BoolOption& option) {
    if (option.hasValue()) {
        return option.getValue();
    }
    return false;
}

//
// PatternBenefit
//

const mlir::PatternBenefit vpux::benefitLow(1);
const mlir::PatternBenefit vpux::benefitMid(2);
const mlir::PatternBenefit vpux::benefitHigh(3);

// Return a pattern benefit vector from large to small
SmallVector<mlir::PatternBenefit> vpux::getBenefitLevels(uint32_t levels) {
    SmallVector<mlir::PatternBenefit> benefitLevels;
    for (const auto level : irange(levels) | reversed) {
        benefitLevels.push_back(mlir::PatternBenefit(level));
    }
    return benefitLevels;
}

//
// FunctionPass
//

void vpux::FunctionPass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::FunctionPass::runOnOperation() {
    if (getOperation().isExternal()) {
        return;
    }

    try {
        _log.trace("Run on Function '{0}'", getOperation().getName());

        _log = _log.nest();
        safeRunOnFunc();
        _log = _log.unnest();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
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

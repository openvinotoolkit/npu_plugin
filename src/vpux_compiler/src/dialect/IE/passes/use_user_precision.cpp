//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// UseUserPrecisionPass
//

class UseUserPrecisionPass final : public IE::UseUserPrecisionBase<UseUserPrecisionPass> {
public:
    explicit UseUserPrecisionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnModule
//

void UseUserPrecisionPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    auto userInputs = netInfo.getInputsInfo();
    auto userOutputs = netInfo.getOutputsInfo();

    const auto funcType = netFunc.getType();

    SmallVector<mlir::Type> newArgTypes(netFunc.getNumArguments());

    for (const auto& p : funcType.getInputs() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = p.value().cast<vpux::NDTypeInterface>();
        const auto userType = userInputs[ind].userType().cast<vpux::NDTypeInterface>();

        const auto newType = origType.changeElemType(userType.getElementType());
        newArgTypes[ind] = newType;
    }

    SmallVector<mlir::Type> newResultTypes(netFunc.getNumResults());

    for (const auto& p : funcType.getResults() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = p.value().cast<vpux::NDTypeInterface>();
        const auto userType = userOutputs[ind].userType().cast<vpux::NDTypeInterface>();

        const auto newType = origType.changeElemType(userType.getElementType());
        newResultTypes[ind] = newType;
    }

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
                                 vpux::NDTypeInterface newType) -> mlir::Operation* {
        return builder.create<IE::ConvertOp>(loc, newType, val, mlir::TypeAttr::get(newType.getElementType()));
    };

    if (mlir::failed(convertFunc(netFunc, newArgTypes, newResultTypes, cvtOpBuilder, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUseUserPrecisionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUseUserPrecisionPass(Logger log) {
    return std::make_unique<UseUserPrecisionPass>(log);
}

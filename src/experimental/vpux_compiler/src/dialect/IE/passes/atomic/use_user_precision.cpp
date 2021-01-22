//
// Copyright 2020 Intel Corporation.
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
    explicit UseUserPrecisionPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnOperation() final;

private:
    void passBody();

private:
    Logger _log;
};

void UseUserPrecisionPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// passBody
//

void UseUserPrecisionPass::passBody() {
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

        const auto origType = p.value().cast<mlir::ShapedType>();
        const auto userType = userInputs[ind].userType().cast<mlir::ShapedType>();

        const auto newType = mlir::RankedTensorType::get(origType.getShape(), userType.getElementType());
        newArgTypes[ind] = newType;
    }

    SmallVector<mlir::Type> newResultTypes(netFunc.getNumResults());

    for (const auto& p : funcType.getResults() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = p.value().cast<mlir::ShapedType>();
        const auto userType = userOutputs[ind].userType().cast<mlir::ShapedType>();

        const auto newType = mlir::RankedTensorType::get(origType.getShape(), userType.getElementType());
        newResultTypes[ind] = newType;
    }

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
                                 mlir::Type newType) -> mlir::Operation* {
        return builder.create<IE::ConvertOp>(loc, val,
                                             mlir::TypeAttr::get(newType.cast<mlir::ShapedType>().getElementType()));
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

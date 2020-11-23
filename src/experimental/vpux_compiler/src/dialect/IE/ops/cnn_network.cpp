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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinOps.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::CNNNetworkOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    auto netFunc = symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, entryPointAttr());

    if (netFunc == nullptr) {
        return printTo(emitError(), "'{0}' entryPoint '@{1}' doesn't refer to existing Function",
                       CNNNetworkOp::getOperationName(), entryPoint());
    }

    auto inputsInfo = to_vector<1>(this->inputsInfo().getOps<DataInfoOp>());
    auto outputsInfo = to_vector<1>(this->outputsInfo().getOps<DataInfoOp>());

    const auto netFuncType = netFunc.getType();

    if (netFuncType.getNumInputs() != inputsInfo.size()) {
        return printTo(emitError(),
                       "'{0}' entryPoint '@{1}' inputs count '{2}' doesn't match "
                       "userInputs count '{3}'",
                       CNNNetworkOp::getOperationName(), entryPoint(), netFuncType.getNumInputs(), inputsInfo.size());
    }

    for (const auto ind : irange(netFuncType.getNumInputs())) {
        const auto inType = netFuncType.getInput(ind).dyn_cast<mlir::RankedTensorType>();

        if (inType == nullptr) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' input #{2} is not a "
                           "'RankedTensor'",
                           CNNNetworkOp::getOperationName(), entryPoint(), ind);
        }

        const auto userLayout = inputsInfo[ind].layout();

        if (getRank(userLayout) != inType.getRank()) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' input #{2} is not compatible "
                           "with userLayout '{3}'",
                           CNNNetworkOp::getOperationName(), entryPoint(), ind, userLayout);
        }
    }

    if (netFuncType.getNumResults() != outputsInfo.size()) {
        return printTo(emitError(),
                       "'{0}' entryPoint '@{1}' outputs count '{2}' doesn't match "
                       "userOutputs count '{3}'",
                       CNNNetworkOp::getOperationName(), entryPoint(), netFuncType.getNumResults(), outputsInfo.size());
    }

    for (const auto ind : irange(netFuncType.getNumResults())) {
        const auto outType = netFuncType.getResult(ind).dyn_cast<mlir::RankedTensorType>();

        if (outType == nullptr) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' output #{2} is not a "
                           "'RankedTensor'",
                           CNNNetworkOp::getOperationName(), entryPoint(), ind);
        }

        const auto userLayout = outputsInfo[ind].layout();

        if (getRank(userLayout) != outType.getRank()) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' output #{2} is not compatible "
                           "with userLayout '{3}'",
                           CNNNetworkOp::getOperationName(), entryPoint(), ind, userLayout);
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::CNNNetworkOp::getFromModule(mlir::ModuleOp module, CNNNetworkOp& netOp,
                                                          mlir::FuncOp& netFunc) {
    auto netOps = to_vector<1>(module.getOps<CNNNetworkOp>());
    if (netOps.size() != 1) {
        return printTo(module.emitError(), "Module {0} doesn't contain IE.{1} Operation", module.getName(),
                       CNNNetworkOp::getOperationName());
    }

    netOp = netOps.front();
    netFunc = module.lookupSymbol<mlir::FuncOp>(netOp.entryPointAttr());

    return mlir::success(netFunc != nullptr);
}

mlir::LogicalResult vpux::IE::verifyOp(CNNNetworkOp op) {
    if (mlir::failed(checkNetworkDataInfoBlock<CNNNetworkOp, DataInfoOp, EndOp>(
                op, op.inputsInfo().front().getOperations(), "inputInfo"))) {
        return mlir::failure();
    }

    if (mlir::failed(checkNetworkDataInfoBlock<CNNNetworkOp, DataInfoOp, EndOp>(
                op, op.outputsInfo().front().getOperations(), "outputsInfo"))) {
        return mlir::failure();
    }

    return mlir::success();
}

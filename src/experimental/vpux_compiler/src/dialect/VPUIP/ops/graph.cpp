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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinDialect.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::GraphOp::verifySymbolUses(
        mlir::SymbolTableCollection& symbolTable) {
    auto netFunc =
            symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this,
                                                              entryPointAttr());

    if (netFunc == nullptr) {
        return printTo(
                emitError(),
                "'{0}' entryPoint '@{1}' doesn't refer to existing Function",
                GraphOp::getOperationName(),
                entryPoint());
    }

    auto inputsInfo = to_vector<1>(this->inputsInfo().getOps<TensorInfoOp>());
    auto outputsInfo = to_vector<1>(this->outputsInfo().getOps<TensorInfoOp>());

    const auto netFuncType = netFunc.getType();

    if (netFuncType.getNumInputs() != inputsInfo.size() + outputsInfo.size()) {
        return printTo(
                emitError(),
                "'{0}' entryPoint '@{1}' inputs count '{2}' doesn't match "
                "userInputs count '{3}' and userOutputs count '{4}'",
                GraphOp::getOperationName(),
                entryPoint(),
                netFuncType.getNumInputs(),
                inputsInfo.size(),
                outputsInfo.size());
    }
    if (netFuncType.getNumResults() != 0) {
        return printTo(emitError(),
                       "'{0}' entryPoint '@{1}' can't have results, got '{2}'",
                       GraphOp::getOperationName(),
                       entryPoint(),
                       netFuncType.getNumResults());
    }

    for (const auto& p : zip(netFuncType.getInputs(),
                             concat<TensorInfoOp>(inputsInfo, outputsInfo)) |
                                 indexed) {
        const auto runtimeType =
                std::get<0>(p.value()).dyn_cast<mlir::MemRefType>();

        if (runtimeType == nullptr) {
            return printTo(
                    emitError(),
                    "'{0}' entryPoint '@{1}' input #{2} is not a 'MemRefType'",
                    GraphOp::getOperationName(),
                    entryPoint(),
                    p.index());
        }

        auto userInfo = std::get<1>(p.value());
        const auto userLayout = userInfo.layout();

        if (checked_cast<unsigned>(runtimeType.getRank()) !=
            userLayout.getNumDims()) {
            return printTo(
                    emitError(),
                    "'{0}' entryPoint '@{1} input #{2} is not compatible "
                    "with {3} '{4}'",
                    GraphOp::getOperationName(),
                    entryPoint(),
                    p.index(),
                    p.index() < inputsInfo.size() ? "user layout"
                                                  : "user layout",
                    userLayout);
        }
    }

    return mlir::success();
}

mlir::LogicalResult
        vpux::VPUIP::GraphOp::getFromModule(mlir::ModuleOp module,
                                            GraphOp& graphOp,
                                            mlir::FuncOp& graphFunc) {
    auto graphOps = to_vector<1>(module.getOps<GraphOp>());
    if (graphOps.size() != 1) {
        return printTo(module.emitError(),
                       "Module {0} doesn't contain VPUIP.{1} Operation",
                       module.getName(),
                       GraphOp::getOperationName());
    }

    graphOp = graphOps.front();
    graphFunc = module.lookupSymbol<mlir::FuncOp>(graphOp.entryPointAttr());

    return mlir::success(graphFunc != nullptr);
}

mlir::LogicalResult vpux::VPUIP::verifyOp(GraphOp op) {
    if (mlir::failed(
                IE::checkNetworkDataInfoBlock<GraphOp, TensorInfoOp, EndOp>(
                        op,
                        op.inputsInfo().front().getOperations(),
                        "inputInfo"))) {
        return mlir::failure();
    }

    if (mlir::failed(
                IE::checkNetworkDataInfoBlock<GraphOp, TensorInfoOp, EndOp>(
                        op,
                        op.outputsInfo().front().getOperations(),
                        "outputsInfo"))) {
        return mlir::failure();
    }

    return mlir::success();
}

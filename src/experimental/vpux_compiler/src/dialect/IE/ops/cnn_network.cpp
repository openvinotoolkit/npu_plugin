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
#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <unordered_set>

using namespace vpux;

//
// CNNNetworkOp
//

mlir::LogicalResult vpux::IE::CNNNetworkOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    auto netFunc = symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, entryPointAttr());

    if (netFunc == nullptr) {
        return printTo(emitError(), "'{0}' entryPoint '@{1}' doesn't refer to existing Function", getOperationName(),
                       entryPoint());
    }

    auto inputsInfo = to_vector<1>(this->inputsInfo().getOps<IE::DataInfoOp>());
    auto outputsInfo = to_vector<1>(this->outputsInfo().getOps<IE::DataInfoOp>());

    const auto netFuncType = netFunc.getType();
    const auto isBufferized = netFuncType.getNumResults() == 0;

    if (isBufferized) {
        if (netFuncType.getNumInputs() != inputsInfo.size() + outputsInfo.size()) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' inputs count '{2}' doesn't match userInputs count '{3}' and "
                           "userOutputs count '{4}'",
                           getOperationName(), entryPoint(), netFuncType.getNumInputs(), inputsInfo.size(),
                           outputsInfo.size());
        }
    } else {
        if (netFuncType.getNumInputs() != inputsInfo.size()) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' inputs count '{2}' doesn't match userInputs count '{3}'",
                           getOperationName(), entryPoint(), netFuncType.getNumInputs(), inputsInfo.size());
        }

        if (netFuncType.getNumResults() != outputsInfo.size()) {
            return printTo(emitError(),
                           "'{0}' entryPoint '@{1}' outputs count '{2}' doesn't match userOutputs count '{3}'",
                           getOperationName(), entryPoint(), netFuncType.getNumResults(), outputsInfo.size());
        }
    }

    for (const auto ind : irange(inputsInfo.size())) {
        const auto funcType = netFuncType.getInput(ind).dyn_cast<mlir::ShapedType>();

        if (funcType == nullptr) {
            return printTo(emitError(), "'{0}' entryPoint '@{1}' input #{2} is not a 'ShapedType'", getOperationName(),
                           entryPoint(), ind);
        }

        const auto userType = inputsInfo[ind].userType().dyn_cast<mlir::ShapedType>();

        if (userType == nullptr) {
            return printTo(emitError(), "'{0}' user input #{1} is not a 'ShapedType'", getOperationName(), ind);
        }

        if (funcType.getNumElements() != userType.getNumElements()) {
            return printTo(emitError(), "'{0}' entryPoint '@{1}' input #{2} is not compatible with user type '{3}'",
                           getOperationName(), entryPoint(), ind, userType);
        }
    }

    for (const auto ind : irange(outputsInfo.size())) {
        const auto funcType = isBufferized ? netFuncType.getInput(inputsInfo.size() + ind).dyn_cast<mlir::ShapedType>()
                                           : netFuncType.getResult(ind).dyn_cast<mlir::ShapedType>();

        if (funcType == nullptr) {
            return printTo(emitError(), "'{0}' entryPoint '@{1}' output #{2} is not a 'ShapedType'", getOperationName(),
                           entryPoint(), ind);
        }

        const auto userType = outputsInfo[ind].userType().dyn_cast<mlir::ShapedType>();

        if (userType == nullptr) {
            return printTo(emitError(), "'{0}' user output #{1} is not a 'ShapedType'", getOperationName(), ind);
        }

        if (funcType.getNumElements() != userType.getNumElements()) {
            return printTo(emitError(), "'{0}' entryPoint '@{1}' output #{2} is not compatible with user type '{3}'",
                           getOperationName(), entryPoint(), ind, userType);
        }
    }

    return mlir::success();
}

namespace {

mlir::LogicalResult verifyDataInfoRegion(mlir::Operation* op, mlir::Region& region, StringRef regionName) {
    if (region.getBlocks().size() != 1) {
        return printTo(op->emitError(), "'{0}' Region for Operation '{1}' must contain exact 1 Block", regionName,
                       op->getName());
    }

    auto& allOps = region.front().getOperations();
    const auto totalNumOps = allOps.size();

    for (auto&& p : allOps | indexed) {
        auto& infoOp = p.value();

        if (static_cast<size_t>(p.index()) == totalNumOps - 1) {
            if (!mlir::isa<IE::EndOp>(infoOp)) {
                return printTo(op->emitError(),
                               "'{0}' Region for Operation '{1}' must end with Terminator '{2}', got '{3}'", regionName,
                               op->getName(), IE::EndOp::getOperationName(), infoOp.getName());
            }
        } else {
            if (!mlir::isa<IE::DataInfoOp>(infoOp)) {
                return printTo(op->emitError(),
                               "'{0}' Region for Operation '{1}' must contain only DataInfo operations, got '{2}'",
                               regionName, op->getName(), infoOp.getName());
            }
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::verifyOp(CNNNetworkOp op) {
    auto entryPointAttr = op.entryPointAttr();

    if (entryPointAttr == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' entryPoint attribute is NULL", op->getName());
    }

    if (mlir::failed(verifyDataInfoRegion(op, op.inputsInfo(), "inputInfo"))) {
        return mlir::failure();
    }
    if (mlir::failed(verifyDataInfoRegion(op, op.outputsInfo(), "outputsInfo"))) {
        return mlir::failure();
    }

    auto inputsInfo = op.getInputsInfo();
    auto outputsInfo = op.getOutputsInfo();

    if (outputsInfo.empty()) {
        return printTo(op->emitError(), "Operation '{0}' has no outputs", op->getName());
    }

    std::unordered_set<StringRef> usedNames;
    for (auto info : concat<IE::DataInfoOp>(inputsInfo, outputsInfo)) {
        const auto res = usedNames.insert(info.name()).second;
        if (!res) {
            return printTo(op->emitError(), "Operation '{0}' has duplicating DataInfo name '{1}'", op->getName(),
                           info.name());
        }
    }

    return mlir::success();
}

SmallVector<IE::DataInfoOp, 1> vpux::IE::CNNNetworkOp::getInputsInfo() {
    return to_vector<1>(inputsInfo().getOps<IE::DataInfoOp>());
}

SmallVector<IE::DataInfoOp, 1> vpux::IE::CNNNetworkOp::getOutputsInfo() {
    return to_vector<1>(outputsInfo().getOps<IE::DataInfoOp>());
}

void vpux::IE::CNNNetworkOp::getFromModule(mlir::ModuleOp module, CNNNetworkOp& netInfo, mlir::FuncOp& netFunc) {
    auto netOps = to_vector<1>(module.getOps<CNNNetworkOp>());

    VPUX_THROW_UNLESS(netOps.size() == 1, "Can't have more than one 'IE::CNNNetworkOp' Operation in Module, got '{0}'",
                      netOps.size());

    netInfo = netOps.front();
    netFunc = module.lookupSymbol<mlir::FuncOp>(netInfo.entryPointAttr());

    VPUX_THROW_UNLESS(netFunc != nullptr, "Can't find entryPoint '@{0}' for '{1}' Operation", netInfo.entryPoint(),
                      netInfo->getName());
}

//
// DataInfoOp
//

mlir::LogicalResult vpux::IE::verifyOp(DataInfoOp op) {
    const auto userType = op.userType().dyn_cast<mlir::MemRefType>();

    if (userType == nullptr) {
        return printTo(op.emitError(), "'{0}' userType is not a 'MemRefType', got '{1}'", op->getName(), userType);
    }

    const auto precision = userType.getElementType();

    if (!(precision.isSignedInteger() || precision.isUnsignedInteger() || precision.isa<mlir::FloatType>())) {
        return printTo(op->emitError(),
                       "Operation '{0}' has unsupported userType precision '{1}', it must be either Float or Integer",
                       op->getName(), precision);
    }

    if (!DimsOrder::fromType(userType).hasValue()) {
        return printTo(op->emitError(), "Operation '{0}' userType '{1}' has unsupported layout", op->getName(),
                       userType);
    }

    return mlir::success();
}

DimsOrder vpux::IE::DataInfoOp::getDimsOrder() {
    return DimsOrder::fromType(userType().cast<mlir::MemRefType>()).getValue();
}

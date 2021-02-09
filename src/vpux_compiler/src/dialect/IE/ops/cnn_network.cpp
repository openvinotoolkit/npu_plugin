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
        return errorAt(*this, "entryPoint '@{0}' doesn't refer to existing Function", entryPoint());
    }

    auto inputsInfo = to_small_vector(this->inputsInfo().getOps<IE::DataInfoOp>());
    auto outputsInfo = to_small_vector(this->outputsInfo().getOps<IE::DataInfoOp>());

    const auto netFuncType = netFunc.getType();
    const auto isBufferized = netFuncType.getNumResults() == 0;

    if (isBufferized) {
        if (netFuncType.getNumInputs() != inputsInfo.size() + outputsInfo.size()) {
            return errorAt(*this,
                           "entryPoint '@{0}' inputs count '{1}' doesn't match userInputs count '{2}' and "
                           "userOutputs count '{3}'",
                           entryPoint(), netFuncType.getNumInputs(), inputsInfo.size(), outputsInfo.size());
        }
    } else {
        if (netFuncType.getNumInputs() != inputsInfo.size()) {
            return errorAt(*this, "entryPoint '@{0}' inputs count '{1}' doesn't match userInputs count '{2}'",
                           entryPoint(), netFuncType.getNumInputs(), inputsInfo.size());
        }

        if (netFuncType.getNumResults() != outputsInfo.size()) {
            return errorAt(*this, "entryPoint '@{0}' outputs count '{1}' doesn't match userOutputs count '{2}'",
                           entryPoint(), netFuncType.getNumResults(), outputsInfo.size());
        }
    }

    for (const auto ind : irange(inputsInfo.size())) {
        const auto funcType = netFuncType.getInput(static_cast<uint32_t>(ind)).dyn_cast<mlir::ShapedType>();

        if (funcType == nullptr) {
            return errorAt(*this, "entryPoint '@{0}' input #{1} is not a 'ShapedType'", entryPoint(), ind);
        }

        const auto userType = inputsInfo[ind].userType().dyn_cast<mlir::ShapedType>();

        if (userType == nullptr) {
            return errorAt(*this, "User input #{0} is not a 'ShapedType'", ind);
        }

        if (funcType.getNumElements() != userType.getNumElements()) {
            return errorAt(*this, "entryPoint '@{0}' input #{1} is not compatible with user type '{2}'", entryPoint(),
                           ind, userType);
        }
    }

    for (const auto ind : irange(outputsInfo.size())) {
        const auto rawInd = inputsInfo.size() + ind;

        const auto funcType = isBufferized
                                      ? netFuncType.getInput(static_cast<uint32_t>(rawInd)).dyn_cast<mlir::ShapedType>()
                                      : netFuncType.getResult(static_cast<uint32_t>(ind)).dyn_cast<mlir::ShapedType>();

        if (funcType == nullptr) {
            return errorAt(*this, "entryPoint '@{0}' output #{1} is not a 'ShapedType'", entryPoint(), ind);
        }

        const auto userType = outputsInfo[ind].userType().dyn_cast<mlir::ShapedType>();

        if (userType == nullptr) {
            return errorAt(*this, "User output #{0} is not a 'ShapedType'", ind);
        }

        if (funcType.getNumElements() != userType.getNumElements()) {
            return errorAt(*this, "entryPoint '@{0}' output #{1} is not compatible with user type '{2}'", entryPoint(),
                           ind, userType);
        }
    }

    return mlir::success();
}

namespace {

mlir::LogicalResult verifyDataInfoRegion(mlir::Operation* op, mlir::Region& region, StringRef regionName) {
    if (region.getBlocks().size() != 1) {
        return errorAt(op, "'{0}' Region must contain exact 1 Block", regionName);
    }

    auto& allOps = region.front().getOperations();
    const auto totalNumOps = allOps.size();

    for (auto&& p : allOps | indexed) {
        auto& infoOp = p.value();

        if (static_cast<size_t>(p.index()) == totalNumOps - 1) {
            if (!mlir::isa<IE::EndOp>(infoOp)) {
                return errorAt(op, "'{0}' Region must end with Terminator '{1}', got '{2}'", regionName,
                               IE::EndOp::getOperationName(), infoOp.getName());
            }
        } else {
            if (!mlir::isa<IE::DataInfoOp>(infoOp)) {
                return errorAt(op, "'{0}' Region must contain only DataInfo operations, got '{1}'", regionName,
                               infoOp.getName());
            }
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::verifyOp(CNNNetworkOp op) {
    auto entryPointAttr = op.entryPointAttr();

    if (entryPointAttr == nullptr) {
        return errorAt(op, "entryPoint attribute is NULL");
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
        return errorAt(op, "Operation has no user outputs information");
    }

    std::unordered_set<StringRef> usedNames;
    for (auto info : concat<IE::DataInfoOp>(inputsInfo, outputsInfo)) {
        const auto res = usedNames.insert(info.name()).second;
        if (!res) {
            return errorAt(op, "Operation has duplicating DataInfo name '{0}'", info.name());
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
    auto netOps = to_small_vector(module.getOps<CNNNetworkOp>());

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
        return errorAt(op, "User type is not a 'MemRefType', got '{0}'", userType);
    }

    const auto precision = userType.getElementType();

    if (!(precision.isSignedInteger() || precision.isUnsignedInteger() || precision.isa<mlir::FloatType>())) {
        return errorAt(op, "Operation has unsupported userType precision '{0}', it must be either Float or Integer",
                       precision);
    }

    return mlir::success();
}

DimsOrder vpux::IE::DataInfoOp::getDimsOrder() {
    return DimsOrder::fromType(userType().cast<mlir::MemRefType>());
}

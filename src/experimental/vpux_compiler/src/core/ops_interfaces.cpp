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

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/SymbolTable.h>

#include <unordered_set>

using namespace vpux;

//
// DataInfoInterface
//

mlir::LogicalResult vpux::details::verifyDataInfo(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyDataInfo");

    auto dataInfo = mlir::dyn_cast<DataInfoInterface>(op);
    if (dataInfo == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' is not a DataInfo", op->getName());
    }

    if (!op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
        return printTo(op->emitError(), "Operation '{0}' is not Isolated", op->getName());
    }

    if (op->getParentOp() == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' has no parent", op->getName());
    }

    if (!mlir::isa<NetInfoInterface>(op->getParentOp())) {
        return printTo(op->emitError(), "Operation '{0}' has wrong parent '{1}' (not a NetInfo)", op->getName(),
                       op->getParentOp()->getName());
    }

    auto precision = dataInfo.precision();

    if (precision == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' precision attribute is NULL", op->getName());
    }

    if (!(precision.isSignedInteger() || precision.isUnsignedInteger() || precision.isa<mlir::FloatType>())) {
        return printTo(op->emitError(),
                       "Operation '{0}' has unsupported precision '{1}', it must be either Float or Integer",
                       op->getName(), precision);
    }

    return mlir::success();
}

//
// NetInfoInterface
//

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
            if (!infoOp.hasTrait<mlir::OpTrait::IsTerminator>()) {
                return printTo(op->emitError(),
                               "'{0}' Region for Operation '{1}' must end with some Terminator, got '{2}'", regionName,
                               op->getName(), infoOp.getName());
            }
        } else {
            if (!mlir::isa<DataInfoInterface>(infoOp)) {
                return printTo(op->emitError(),
                               "'{0}' Region for Operation '{1}' must contain only DataInfo operations, got '{2}'",
                               regionName, op->getName(), infoOp.getName());
            }
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::details::verifyNetInfo(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyNetInfo");

    auto netInfo = mlir::dyn_cast<NetInfoInterface>(op);
    if (netInfo == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' is not a NetInfo", op->getName());
    }

    if (!op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
        return printTo(op->emitError(), "Operation '{0}' is not Isolated", op->getName());
    }

    if (!op->hasTrait<mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl>()) {
        return printTo(op->emitError(), "Operation '{0}' is not attached to Module", op->getName());
    }

    if (op->getRegions().size() != 2) {
        return printTo(op->emitError(), "Operation '{0}' must have 2 Regions with inputs/outputs information",
                       op->getName());
    }

    if (!op->hasTrait<mlir::OpTrait::NoRegionArguments>()) {
        return printTo(op->emitError(), "Operation '{0}' Regions must have no arguments", op->getName());
    }

    if (mlir::dyn_cast<mlir::SymbolUserOpInterface>(op) == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' is not a Symbol User", op->getName());
    }

    auto entryPointAttr = netInfo.entryPointAttr();

    if (entryPointAttr == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' entryPoint attribute is NULL", op->getName());
    }

    if (mlir::failed(verifyDataInfoRegion(op, netInfo.inputsInfo(), "inputInfo"))) {
        return mlir::failure();
    }
    if (mlir::failed(verifyDataInfoRegion(op, netInfo.outputsInfo(), "outputsInfo"))) {
        return mlir::failure();
    }

    auto inputsInfo = netInfo.getInputsInfo();
    auto outputsInfo = netInfo.getOutputsInfo();

    if (inputsInfo.empty()) {
        return printTo(op->emitError(), "Operation '{0}' has no inputs information", op->getName());
    }
    if (outputsInfo.empty()) {
        return printTo(op->emitError(), "Operation '{0}' has no outputs information", op->getName());
    }

    std::unordered_set<StringRef> usedNames;
    for (auto info : concat<DataInfoInterface>(inputsInfo, outputsInfo)) {
        const auto res = usedNames.insert(info.name()).second;
        if (!res) {
            return printTo(op->emitError(), "Operation '{0}' has duplicating DataInfo name '{1}'", op->getName(),
                           info.name());
        }
    }

    return mlir::success();
}

mlir::FailureOr<std::pair<mlir::Operation*, mlir::FuncOp>> vpux::details::getNetInfo(mlir::ModuleOp module) {
    auto netOps = to_vector<1>(module.getOps<NetInfoInterface>());
    if (netOps.size() != 1) {
        return mlir::LogicalResult(printTo(module.emitError(), "Module doesn't contain NetInfo Operation"));
    }

    auto netInfo = netOps.front();
    auto netFunc = module.lookupSymbol<mlir::FuncOp>(netInfo.entryPointAttr());

    if (netFunc == nullptr) {
        return mlir::LogicalResult(
                printTo(module.emitError(), "Can't find NetInfo entryPoint '@{0}'", netInfo.entryPoint()));
    }

    return std::make_pair(netInfo.getOperation(), netFunc);
}

SmallVector<DataInfoInterface, 1> vpux::details::getDataInfoVec(mlir::Region& region) {
    return to_vector<1>(region.getOps<DataInfoInterface>());
}

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.cpp.inc>

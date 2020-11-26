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
// LayerInterface
//

mlir::LogicalResult vpux::details::verifyLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyLayer");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' is not a Layer", op->getName());
    }

    // NOTE: `to_vector` is used to overcome `concat` limitations for iterators, which returns copies instead of
    // references
    auto inputs = to_vector<4>(layer.getInputs());
    auto outputs = to_vector<1>(layer.getOutputs());

    if (outputs.empty()) {
        return printTo(op->emitError(), "Layer Operation '{0}' has no outputs", op->getName());
    }

    for (auto var : concat<mlir::Value>(inputs, outputs)) {
        auto type = var.getType();

        if (!type.isa<mlir::ShapedType>()) {
            return printTo(op->emitError(),
                           "Layer Operation '{0}' has input/output with wrong Type, expected ShapedType, got '{1}'",
                           op->getName(), type);
        }
    }

    return mlir::success();
}

//
// ConvertLayerInterface
//

namespace {

template <class ConcreteLayer>
mlir::FailureOr<std::pair<LayerInterface, ConcreteLayer>> getLayer(mlir::Operation* op, StringRef comment) {
    auto base = mlir::dyn_cast<LayerInterface>(op);
    if (base == nullptr) {
        return mlir::LogicalResult(printTo(op->emitError(), "Operation '{0}' is not a Layer", op->getName()));
    }

    auto actual = mlir::dyn_cast<ConcreteLayer>(op);
    if (actual == nullptr) {
        return mlir::LogicalResult(
                printTo(op->emitError(), "Operation '{0}' is not a {1} Layer", op->getName(), comment));
    }

    return std::make_pair(base, actual);
}

mlir::LogicalResult verifyLayerInputsOutputs(LayerInterface layer, size_t numInputs, size_t numOutputs,
                                             StringRef comment) {
    if (layer.getInputs().size() != numInputs) {
        return printTo(layer.emitError(), "{0} Layer '{1}' has wrong number of inputs '{2}', expected '{3}'", comment,
                       layer.getOperation()->getName(), layer.getInputs().size(), numInputs);
    }
    if (layer.getOutputs().size() != numOutputs) {
        return printTo(layer.emitError(), "{0} Layer '{1}' has wrong number of outputs '{2}', expected '{3}'", comment,
                       layer.getOperation()->getName(), layer.getOutputs().size(), numOutputs);
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::details::verifyConvertLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyConvertLayer");

    auto res = getLayer<ConvertLayerInterface>(op, "Convert");
    if (mlir::failed(res)) {
        return mlir::failure();
    }

    auto layer = res->first;
    auto convert = res->second;

    if (mlir::failed(verifyLayerInputsOutputs(layer, 1, 1, "Convert"))) {
        return mlir::failure();
    }

    auto srcType = convert.getSrcType();
    auto dstType = convert.getDstType();

    if (srcType.getShape() != dstType.getShape()) {
        return printTo(op->emitError(), "Convert Layer '{0}' has different shapes for input ('{1}') and output ('{2}')",
                       op->getName(), srcType.getShape(), dstType.getShape());
    }

    return mlir::success();
}

//
// SoftMaxLayerInterface
//

mlir::LogicalResult vpux::details::verifySoftMaxLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySoftMaxLayer");

    auto res = getLayer<SoftMaxLayerInterface>(op, "SoftMax");
    if (mlir::failed(res)) {
        return mlir::failure();
    }

    auto layer = res->first;
    auto softMax = res->second;

    if (mlir::failed(verifyLayerInputsOutputs(layer, 1, 1, "SoftMax"))) {
        return mlir::failure();
    }

    auto srcType = softMax.getSrcType();
    auto dstType = softMax.getDstType();

    if (srcType.getShape() != dstType.getShape()) {
        return printTo(op->emitError(), "SoftMax Layer '{0}' has different shapes for input ('{1}') and output ('{2}')",
                       op->getName(), srcType.getShape(), dstType.getShape());
    }

    if (srcType.getElementType() != dstType.getElementType()) {
        return printTo(op->emitError(),
                       "SoftMax Layer '{0}' has different element type for input ('{1}') and output ('{2}')",
                       op->getName(), srcType.getElementType(), dstType.getElementType());
    }

    const auto workRank = srcType.getShape().size();
    const auto axisInd = softMax.getAxisDim().ind();

    if (axisInd < 0 || checked_cast<size_t>(axisInd) >= workRank) {
        return printTo(op->emitError(), "SoftMax Layer '{0}' axis index '{1}' is out of working rank '{2}'",
                       op->getName(), axisInd, workRank);
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.cpp.inc>

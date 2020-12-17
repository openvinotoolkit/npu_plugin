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

#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>

#include <unordered_set>

using namespace vpux;

//
// LayerInterface
//

mlir::LogicalResult vpux::verifyLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyLayer");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' is not a Layer", op->getName());
    }

    if (op->getOperands().empty()) {
        return printTo(op->emitError(), "Layer Operation '{0}' has no operands", op->getName());
    }

    bool isTensorLayer = false;
    bool isMemRefLayer = false;

    for (auto arg : op->getOperands()) {
        auto type = arg.getType();

        if (type.isa<mlir::RankedTensorType>() || type.isa<mlir::shape::ShapeType>()) {
            if (isMemRefLayer) {
                return printTo(op->emitError(), "Layer Operation '{0}' has a mix of Tensor/Shape and MemRef types",
                               op->getName());
            }

            isTensorLayer = true;
        } else if (type.isa<mlir::MemRefType>()) {
            if (isTensorLayer) {
                return printTo(op->emitError(), "Layer Operation '{0}' has a mix of Tensor/Shape and MemRef types",
                               op->getName());
            }

            isMemRefLayer = true;
        }
    }

    if (!isTensorLayer && !isMemRefLayer) {
        return printTo(op->emitError(), "Layer Operation '{0}' has no Tensor/Shape or MemRef types operands",
                       op->getName());
    }

    for (auto res : op->getResults()) {
        auto type = res.getType();

        if (type.isa<mlir::RankedTensorType>() || type.isa<mlir::shape::ShapeType>()) {
            if (isMemRefLayer) {
                return printTo(op->emitError(), "Layer Operation '{0}' has a mix of Tensor/Shape and MemRef types",
                               op->getName());
            }
        } else if (type.isa<mlir::MemRefType>()) {
            if (isTensorLayer) {
                return printTo(op->emitError(), "Layer Operation '{0}' has a mix of Tensor/Shape and MemRef types",
                               op->getName());
            }

            return printTo(op->emitError(), "Layer Operation '{0}' can return MemRef types", op->getName());
        }
    }

    auto inputs = layer.getInputs();
    auto outputs = layer.getOutputs();

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

mlir::LogicalResult vpux::verifyConvertLayer(mlir::Operation* op) {
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

    auto inputType = convert.inputType();
    auto outputType = convert.outputType();

    if (inputType.getShape() != outputType.getShape()) {
        return printTo(op->emitError(), "Convert Layer '{0}' has different shapes for input ('{1}') and output ('{2}')",
                       op->getName(), inputType.getShape(), outputType.getShape());
    }

    return mlir::success();
}

//
// SoftMaxLayerInterface
//

mlir::LogicalResult vpux::verifySoftMaxLayer(mlir::Operation* op) {
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

    auto inputType = softMax.inputType();
    auto outputType = softMax.outputType();

    if (inputType.getShape() != outputType.getShape()) {
        return printTo(op->emitError(), "SoftMax Layer '{0}' has different shapes for input ('{1}') and output ('{2}')",
                       op->getName(), inputType.getShape(), outputType.getShape());
    }

    if (inputType.getElementType() != outputType.getElementType()) {
        return printTo(op->emitError(),
                       "SoftMax Layer '{0}' has different element type for input ('{1}') and output ('{2}')",
                       op->getName(), inputType.getElementType(), outputType.getElementType());
    }

    const auto workRank = inputType.getShape().size();
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

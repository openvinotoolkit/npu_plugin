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
// ConstantInterface
//

mlir::LogicalResult vpux::verifyConstant(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyConstant");

    if (!op->hasTrait<mlir::OpTrait::ConstantLike>()) {
        return errorAt(op, "Operation '{0}' is not a ConstantLike", op->getName());
    }

    auto constant = mlir::dyn_cast<ConstantInterface>(op);
    if (constant == nullptr) {
        return errorAt(op, "Operation '{0}' is not a Constant", op->getName());
    }

    const auto contentType = constant.getContentType();
    const auto actualType = constant.getActualType();

    if (!contentType.hasStaticShape()) {
        return errorAt(op, "Can't use dynamic shape for constant content");
    }
    if (!actualType.hasStaticShape()) {
        return errorAt(op, "Can't use dynamic shape for constant result value");
    }

    if (contentType.getNumElements() != actualType.getNumElements()) {
        return errorAt(op, "Content type '{0}' and actual type '{1}' are not compatible", contentType, actualType);
    }

    return mlir::success();
}

//
// LayerInterface
//

mlir::LogicalResult vpux::verifyLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyLayer");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' is not a Layer", op->getName());
    }

    if (op->getOperands().empty()) {
        return errorAt(op, "Layer Operation has no operands");
    }

    bool isTensorLayer = false;
    bool isMemRefLayer = false;

    for (auto arg : op->getOperands()) {
        auto type = arg.getType();

        if (type.isa<mlir::RankedTensorType>() || type.isa<mlir::shape::ShapeType>()) {
            if (isMemRefLayer) {
                return errorAt(op, "Layer Operation has a mix of Tensor/Shape and MemRef types");
            }

            isTensorLayer = true;
        } else if (type.isa<mlir::MemRefType>()) {
            if (isTensorLayer) {
                return errorAt(op, "Layer Operation has a mix of Tensor/Shape and MemRef types");
            }

            isMemRefLayer = true;
        }
    }

    if (!isTensorLayer && !isMemRefLayer) {
        return errorAt(op, "Layer Operation has no Tensor/Shape or MemRef types operands");
    }

    for (auto res : op->getResults()) {
        auto type = res.getType();

        if (type.isa<mlir::RankedTensorType>() || type.isa<mlir::shape::ShapeType>()) {
            if (isMemRefLayer) {
                return errorAt(op, "Layer Operation has a mix of Tensor/Shape and MemRef types");
            }
        } else if (type.isa<mlir::MemRefType>()) {
            if (isTensorLayer) {
                return errorAt(op, "Layer Operation has a mix of Tensor/Shape and MemRef types");
            }

            return errorAt(op, "Layer Operation can't return MemRef types");
        }
    }

    auto inputs = layer.getInputs();
    auto outputs = layer.getOutputs();

    if (outputs.empty()) {
        return errorAt(op, "Layer Operation has no outputs");
    }

    for (auto var : concat<mlir::Value>(inputs, outputs)) {
        auto type = var.getType();

        if (!type.isa<mlir::ShapedType>()) {
            return errorAt(op, "Layer Operation has input/output with wrong Type, expected ShapedType, got '{0}'",
                           type);
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
        return errorAt(op, "Operation is not a Layer");
    }

    auto actual = mlir::dyn_cast<ConcreteLayer>(op);
    if (actual == nullptr) {
        return errorAt(op, "Operation is not a {0} Layer", comment);
    }

    return std::make_pair(base, actual);
}

mlir::LogicalResult verifyLayerInputsOutputs(LayerInterface layer, size_t numInputs, size_t numOutputs,
                                             StringRef comment) {
    if (layer.getInputs().size() != numInputs) {
        return errorAt(layer, "{0} Layer has wrong number of inputs '{1}', expected '{2}'", comment,
                       layer.getInputs().size(), numInputs);
    }
    if (layer.getOutputs().size() != numOutputs) {
        return errorAt(layer, "{0} Layer has wrong number of outputs '{1}', expected '{2}'", comment,
                       layer.getOutputs().size(), numOutputs);
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
        return errorAt(op, "Convert Layer has different shapes for input '{0}' and output '{1}'", inputType.getShape(),
                       outputType.getShape());
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
        return errorAt(op, "SoftMax Layer has different shapes for input '{0}' and output '{1}'", inputType.getShape(),
                       outputType.getShape());
    }

    if (inputType.getElementType() != outputType.getElementType()) {
        return errorAt(op, "SoftMax Layer has different element type for input '{0}' and output '{1}'",
                       inputType.getElementType(), outputType.getElementType());
    }

    const auto workRank = inputType.getShape().size();
    const auto axisInd = softMax.getAxisDim().ind();

    if (axisInd < 0 || checked_cast<size_t>(axisInd) >= workRank) {
        return errorAt(op, "SoftMax Layer axis index '{0}' is out of working rank '{1}'", axisInd, workRank);
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.cpp.inc>

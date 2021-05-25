//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

namespace {

// Returns the number of operands that are the result of other layers
// For this ops:
// %6 = VPUIP.SomeTaskUPA inputs(%1 : memref, %2 : memref) outputs(%3 : memref) waits(%4 : !VPUIP.Barrier) updates(%5 :
// !VPUIP.Barrier)) numOperands() == 5 <==> %1, %2, %3, %4, %5 getLastMemRefPosition() == 3  <==> %1, %2 and %3
ptrdiff_t getLastMemRefPosition(const mlir::ValueRange& vals) {
    return std::find_if(vals.begin(), vals.end(),
                        [](mlir::Value val) {
                            return !val.getType().isa<mlir::MemRefType>();
                        }) -
           vals.begin();
}

}  // namespace

//
// DataOrderInfo
//

void DataOrderInfo::printFormat(llvm::raw_ostream& stream) const {
    stream << "Order info [";
    for (size_t i = 0; i < _inputOrders.size(); ++i) {
        stream << " inL[" << i << "]=";
        if (_inputOrders[i].hasValue()) {
            _inputOrders[i]->printFormat(stream);
        } else {
            stream << "ANY";
        }
    }

    stream << ";";
    for (size_t i = 0; i < _outputOrders.size(); ++i) {
        stream << " outL[" << i << "]=";
        if (_outputOrders[i].hasValue()) {
            _outputOrders[i]->printFormat(stream);
        } else {
            stream << "ANY";
        }
    }

    stream << " ]";
}

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
        }
    }

    if (layer.getOutputs().empty()) {
        return errorAt(op, "Layer Operation has no outputs");
    }

    for (auto& var : layer.getOpOperands()) {
        auto type = var.get().getType();

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

mlir::LogicalResult vpux::verifyConvertLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyConvertLayer");

    auto convert = mlir::dyn_cast<ConvertLayerInterface>(op);
    if (convert == nullptr) {
        return errorAt(op, "Operation is not a Convert Layer");
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

    auto softMax = mlir::dyn_cast<SoftMaxLayerInterface>(op);
    if (softMax == nullptr) {
        return errorAt(op, "Operation is not a SoftMax Layer");
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
// RTLayer
//

mlir::LogicalResult vpux::verifyRTLayerOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyRTLayerOp");

    if (!mlir::isa<LayerInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not a Layer", op->getName());
    }

    auto hasTensor = llvm::any_of(op->getOperands(), [](mlir::Value type) {
        return type.getType().isa<mlir::RankedTensorType>();
    });

    hasTensor &= llvm::any_of(op->getResults(), [](mlir::Value type) {
        return type.getType().isa<mlir::RankedTensorType>();
    });

    if (hasTensor) {
        return errorAt(op, "Operation '{0}' is not a RT Layer, it operates with Tensor types", op->getName());
    }

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    if (inNum < outNum) {
        return errorAt(op,
                       "The number of operands must always be greater than or equal to the number of results, since "
                       "they include buffers for the results. inNum={0}; outNum={1}",
                       inNum, outNum);
    }

    return mlir::success();
}

mlir::OperandRange vpux::getRTLayerInOperand(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerInputs");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().take_front(inNum - outNum);
}

mlir::OperandRange vpux::getRTLayerOutOperand(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerInputs");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().slice(inNum - outNum, outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::getRTLayerInOpOperands(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerInputs");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().take_front(inNum - outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::getRTLayerOutOpOperands(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerInputs");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().slice(inNum - outNum, outNum);
}

DataOrderInfo vpux::getRTLayerDataOrderInfo(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getRTLayerDataOrderInfo");

    const auto inputs = getRTLayerInOperand(op);
    const auto outputs = getRTLayerOutOperand(op);

    DataOrderInfo orderInfo{inputs.size(), outputs.size()};

    for (const auto& val : inputs | indexed) {
        orderInfo.setInput(val.index(), DimsOrder::fromValue(val.value()));
    }

    for (const auto& val : outputs | indexed) {
        orderInfo.setOutput(val.index(), DimsOrder::fromValue(val.value()));
    }

    return orderInfo;
}

mlir::Value vpux::getRTLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getRTLayerViewSource");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    VPUX_THROW_UNLESS(resultInd < outNum, "Result index '{0}' is out of range '{1}'", resultInd, outNum);
    return op->getOperand(checked_cast<unsigned>(inNum - outNum + resultInd));
}

mlir::LogicalResult vpux::inferRTLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                                  SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inNum = getLastMemRefPosition(operands);

    VPUX_THROW_UNLESS(numResults < checked_cast<size_t>(inNum),
                      "Call inferRTLayerReturnTypes for non RT Layer Operation");

    inferredReturnTypes.reserve(numResults);
    for (const auto val : operands.slice(inNum - numResults, numResults)) {
        inferredReturnTypes.push_back(val.getType());
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.cpp.inc>

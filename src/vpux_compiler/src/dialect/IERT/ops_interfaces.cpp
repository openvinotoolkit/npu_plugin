//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// LayerOpInterface
//

namespace {

// Returns the number of operands that are the result of other layers
// For this ops:
// %6 = VPUIP.SomeTaskUPA inputs(%1 : memref, %2 : memref) outputs(%3 : memref) waits(%4 : !VPUIP.Barrier) updates(%5 :
// !VPUIP.Barrier)) numOperands() == 5 <==> %1, %2, %3, %4, %5 getLastMemRefPosition() == 3  <==> %1, %2 and %3
ptrdiff_t getLastMemRefPosition(mlir::ValueRange vals) {
    return std::find_if(vals.begin(), vals.end(),
                        [](mlir::Value val) {
                            return !val.getType().isa<mlir::MemRefType>();
                        }) -
           vals.begin();
}

}  // namespace

mlir::LogicalResult vpux::IERT::verifyLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyLayer");

    if (op->getOperands().empty()) {
        return errorAt(op, "RunTime Layer Operation has no operands");
    }
    if (op->getResults().empty()) {
        return errorAt(op, "RunTime Layer Operation has no results");
    }

    for (auto& arg : op->getOpOperands()) {
        const auto type = arg.get().getType();

        if (type.isa<mlir::RankedTensorType>()) {
            return errorAt(op, "RunTime Layer Operation has Tensor operand #{0}", arg.getOperandNumber());
        }
    }
    for (auto res : op->getOpResults()) {
        const auto type = res.getType();

        if (type.isa<mlir::RankedTensorType>()) {
            return errorAt(op, "RunTime Layer Operation has Tensor result #{0}", res.getResultNumber());
        }
    }

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    if (inNum < outNum) {
        return errorAt(op,
                       "The number of operands must always be greater than or equal to the number of results, since "
                       "they include buffers for the results : inNum={0} outNum={1}",
                       inNum, outNum);
    }

    return mlir::success();
}

mlir::OperandRange vpux::IERT::getLayerInputs(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerInputs");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().take_front(inNum - outNum);
}

mlir::OperandRange vpux::IERT::getLayerOutputs(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerOutputs");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOperands().slice(inNum - outNum, outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::IERT::getLayerInOpOperands(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerInOpOperands");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().take_front(inNum - outNum);
}

MutableArrayRef<mlir::OpOperand> vpux::IERT::getLayerOutOpOperands(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerOutOpOperands");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    return op->getOpOperands().slice(inNum - outNum, outNum);
}

IE::DataOrderInfo vpux::IERT::getLayerDataOrderInfo(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerDataOrderInfo");

    const auto inputs = getLayerInputs(op);
    const auto outputs = getLayerOutputs(op);

    IE::DataOrderInfo orderInfo{inputs.size(), outputs.size()};

    for (const auto& val : inputs | indexed) {
        orderInfo.setInput(val.index(), DimsOrder::fromValue(val.value()));
    }

    for (const auto& val : outputs | indexed) {
        orderInfo.setOutput(val.index(), DimsOrder::fromValue(val.value()));
    }

    return orderInfo;
}

mlir::Value vpux::IERT::getLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerViewSource");

    const auto inNum = getLastMemRefPosition(op->getOperands());
    const auto outNum = getLastMemRefPosition(op->getResults());

    VPUX_THROW_UNLESS(resultInd < outNum, "Result index '{0}' is out of range '{1}'", resultInd, outNum);
    return op->getOperand(checked_cast<unsigned>(inNum - outNum + resultInd));
}

mlir::LogicalResult vpux::IERT::inferLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                                      SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inNum = getLastMemRefPosition(operands);

    VPUX_THROW_UNLESS(numResults < checked_cast<size_t>(inNum),
                      "Call inferLayerReturnTypes for non RT Layer Operation");

    inferredReturnTypes.reserve(numResults);
    for (const auto val : operands.slice(inNum - numResults, numResults)) {
        inferredReturnTypes.push_back(val.getType());
    }

    return mlir::success();
}

//
// SameShape
//

mlir::LogicalResult vpux::IERT::verifySameShape(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameShape");

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainShape = getShape(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());

        if (shape != mainShape) {
            return errorAt(op, "Operation's input/output shapes mismatch");
        }
    }

    return mlir::success();
}

//
// SameElementType
//

mlir::LogicalResult vpux::IERT::verifySameElementType(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameElementType");

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainElemType = firstInput.getType().cast<mlir::ShapedType>().getElementType();

    for (const auto& val : layer.getOpOperands()) {
        const auto elemType = val.get().getType().cast<mlir::ShapedType>().getElementType();

        if (elemType != mainElemType) {
            return errorAt(op, "Operation's input/output element types mismatch");
        }
    }

    return mlir::success();
}

//
// SameDimsOrder
//

mlir::LogicalResult vpux::IERT::verifySameDimsOrder(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameDimsOrder");

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainOrder = DimsOrder::fromValue(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto order = DimsOrder::fromValue(val.get());

        if (order != mainOrder) {
            return errorAt(op, "Operation's input/output layout mismatch");
        }
    }

    return mlir::success();
}

bool vpux::IERT::isSupportedLayoutSameDimsOrder(mlir::Operation* op, IE::DataOrderInfo& info) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in isSupportedLayoutSameDimsOrder");

    auto layer = mlir::dyn_cast<IE::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' doesn't implement Layer interface", op->getName());

    const auto inputs = layer.getInputs();
    const auto outputs = layer.getOutputs();

    const auto inNum = inputs.size();
    const auto outNum = outputs.size();

    const auto mainOrder = info.hasInput(0) ? info.getInput(0) : DimsOrder::fromValue(inputs[0]);

    for (size_t i = 0; i < inNum; ++i) {
        if (!info.hasInput(i) || info.getInput(i) != mainOrder) {
            IE::fillDataInfo(info, inNum, outNum, mainOrder);
            return false;
        }
    }

    for (size_t i = 0; i < outNum; ++i) {
        if (!info.hasOutput(i) || info.getOutput(i) != mainOrder) {
            IE::fillDataInfo(info, inNum, outNum, mainOrder);
            return false;
        }
    }

    return true;
}

//
// SameInOutDimsOrder
//

mlir::LogicalResult vpux::IERT::verifySameInOutDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} is not layer", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != outOrder) {
        return errorAt(op->getLoc(), "Operation must have the same input and output order. inL={0}, outL={1}", inOrder,
                       outOrder);
    }

    return mlir::success();
}

bool vpux::IERT::isSupportedLayoutSameInOutDimsOrder(mlir::Operation* op, IE::DataOrderInfo& info) {
    auto layer = mlir::dyn_cast<IE::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} is not layer", op->getName());

    if (!info.hasInput(0) || !info.hasOutput(0)) {
        const auto intType = layer.getInputs()[0].getType();
        const auto supportedOrder = info.hasInput(0)
                                            ? info.getInput(0)
                                            : DimsOrder::fromNumDims(intType.cast<mlir::ShapedType>().getRank());

        IE::fillDataInfo(info, 1, 1, supportedOrder);
        return false;
    }

    if (info.getInput(0) != info.getOutput(0)) {
        info.setOutput(0, info.getInput(0));
        return false;
    }

    return true;
}

//
// SameInOutSpecificDimsOrder
//

const std::array<DimsOrder, 2> vpux::IERT::NCHW_NHWC = {DimsOrder::NCHW, DimsOrder::NHWC};
const std::array<DimsOrder, 4> vpux::IERT::CHW_HWC_NCHW_NHWC = {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW,
                                                                DimsOrder::NHWC};

mlir::LogicalResult vpux::IERT::verifySameInOutSpecificDimsOrder(mlir::Operation* op,
                                                                 ArrayRef<DimsOrder> supportedLayouts) {
    if (verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<IERT::LayerOpInterface>(op);

    const auto input = layerOp.getInputs()[0];
    const auto inOrder = DimsOrder::fromValue(input);

    const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
    if (!isSupported) {
        return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
    }

    return mlir::success();
}

bool vpux::IERT::isSupportedLayoutSameInOutSpecificDimsOrder(mlir::Operation* op, IE::DataOrderInfo& info,
                                                             ArrayRef<DimsOrder> supportedLayouts) {
    auto layer = mlir::dyn_cast<IE::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' doesn't implement Layer interface", op->getName());

    const auto intType = layer.getInputs()[0].getType().cast<mlir::ShapedType>();
    const auto defaultOrderIt =
            std::find_if(supportedLayouts.begin(), supportedLayouts.end(), [intType](DimsOrder order) {
                return static_cast<int64_t>(order.numDims()) == intType.getRank();
            });

    VPUX_THROW_UNLESS(defaultOrderIt != supportedLayouts.end(),
                      "Layouts supported ({0}) by the operation '{1}' do not match the rank {2} of the input shape ",
                      supportedLayouts, op->getName(), intType.getRank());

    const auto defaultOrder = *defaultOrderIt;
    if (!info.hasInput(0)) {
        IE::fillDataInfo(info, 1, 1, defaultOrder);
        return false;
    }

    const auto mainOrder = info.getInput(0);
    const auto isSupportedLayout = std::count(supportedLayouts.begin(), supportedLayouts.end(), mainOrder);
    if (isSupportedLayout) {
        return isSupportedLayoutSameInOutDimsOrder(op, info);
    }

    IE::fillDataInfo(info, 1, 1, defaultOrder);
    return false;
}

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/ops_interfaces.cpp.inc>

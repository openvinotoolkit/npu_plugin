//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp>

using namespace vpux;

//
// LayerOpInterface
//

mlir::LogicalResult vpux::IE::verifyLayer(mlir::Operation* op) {
    if (op->getOperands().empty()) {
        return errorAt(op, "Layer Operation has no operands");
    }
    if (op->getResults().empty()) {
        return errorAt(op, "Layer Operation has no results");
    }

    const auto verifyType = [&](mlir::Type type, StringRef name, unsigned ind) {
        if (type.isa<mlir::MemRefType>()) {
            return errorAt(op, "Layer Operation has MemRef {0} #{1}", name, ind);
        }

        if (auto mainType = type.dyn_cast<mlir::ShapedType>()) {
            if (validateQuantElemType(op->getLoc(), mainType).failed()) {
                return mlir::failure();
            }
        }

        return mlir::success();
    };

    for (auto& arg : op->getOpOperands()) {
        if (verifyType(arg.get().getType(), "operand", arg.getOperandNumber()).failed()) {
            return mlir::failure();
        }
    }
    for (auto res : op->getOpResults()) {
        if (verifyType(res.getType(), "result", res.getResultNumber()).failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                               Optional<mlir::Location> loc, mlir::ValueRange operands,
                                               mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                               SmallVectorImpl<mlir::Type>& inferredTypes) {
    SmallVector<mlir::ShapedTypeComponents> components;
    if (mlir::failed(componentsCb(ctx, loc, operands, attrs, regions, components))) {
        return mlir::failure();
    }

    for (const auto& desc : components) {
        VPUX_THROW_UNLESS(desc.hasRank(), "Unranked TensorType is not supported");

        const auto type = mlir::RankedTensorType::get(desc.getDims(), desc.getElementType(), desc.getAttribute());
        inferredTypes.push_back(type);
    }

    return mlir::success();
}

bool vpux::IE::areTypesCompatible(mlir::TypeRange lhs, mlir::TypeRange rhs, IE::TypeComparisonMode elemComparisonModes,
                                  bool checkDimsOrder, bool checkMemSpace) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto p : zip(lhs, rhs)) {
        auto lhsOrigType = std::get<0>(p);
        auto rhsOrigType = std::get<1>(p);

        if (lhsOrigType.getTypeID() != rhsOrigType.getTypeID()) {
            return false;
        }

        auto lhsType = lhsOrigType.dyn_cast<NDTypeInterface>();
        auto rhsType = lhsOrigType.dyn_cast<NDTypeInterface>();

        if (lhsType == nullptr || rhsType == nullptr) {
            return false;
        }

        if (lhsType.getShape() != rhsType.getShape()) {
            return false;
        }

        if (lhsType.getElementType() != rhsType.getElementType()) {
            if (IE::bitEnumContains(elemComparisonModes, IE::TypeComparisonMode::STRICT_EQUAL)) {
                return false;
            }

            const auto lhsQuantizedType = lhsType.getElementType().dyn_cast<mlir::quant::QuantizedType>();
            const auto rhsQuantizedType = rhsType.getElementType().dyn_cast<mlir::quant::QuantizedType>();

            if (!lhsQuantizedType && !rhsQuantizedType) {
                return false;
            } else if (lhsQuantizedType && rhsQuantizedType) {
                if ((lhsQuantizedType.getExpressedType() != rhsQuantizedType.getExpressedType()) ||
                    (lhsQuantizedType.getStorageType() != rhsQuantizedType.getStorageType())) {
                    if (!IE::bitEnumContains(elemComparisonModes, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
                        return false;
                    }
                }
            } else {
                if (!IE::bitEnumContains(elemComparisonModes, IE::TypeComparisonMode::ALLOW_QUANT_MIXED_PRECISION)) {
                    return false;
                }
            }
        }

        if (checkDimsOrder) {
            const auto order1 = lhsType.getDimsOrder();
            const auto order2 = rhsType.getDimsOrder();

            if (order1 != order2) {
                return false;
            }
        }

        if (checkMemSpace) {
            const auto memSpace1 = lhsType.getMemSpace();
            const auto memSpace2 = rhsType.getMemSpace();

            if (memSpace1 != memSpace2) {
                return false;
            }
        }
    }

    return true;
}

//
// LayoutInfoOpInterface
//

void vpux::IE::LayerLayoutInfo::setInput(size_t ind, const DimsOrder& info) {
    const auto prevInfo = getInput(ind);
    VPUX_THROW_UNLESS(info.numDims() == prevInfo.numDims(), "New order '{0}' doesn't match original rank '{1}'", info,
                      prevInfo.numDims());

    LayerDataInfo<DimsOrder>::setInput(ind, info);
}

void vpux::IE::LayerLayoutInfo::setOutput(size_t ind, const DimsOrder& info) {
    const auto prevInfo = getOutput(ind);
    VPUX_THROW_UNLESS(info.numDims() == prevInfo.numDims(), "New order '{0}' doesn't match original rank '{1}'", info,
                      prevInfo.numDims());

    LayerDataInfo<DimsOrder>::setOutput(ind, info);
}

IE::LayerLayoutInfo vpux::IE::getLayoutInfo(mlir::Operation* op) {
    SmallVector<DimsOrder> inputOrders;
    inputOrders.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputOrders.push_back(DimsOrder::fromValue(val));
    }

    SmallVector<DimsOrder> outputOrders;
    outputOrders.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputOrders.push_back(DimsOrder::fromValue(val));
    }

    return IE::LayerLayoutInfo(std::move(inputOrders), std::move(outputOrders));
}

void vpux::IE::fillDefaultLayoutInfo(IE::LayerLayoutInfo& info) {
    for (auto i : irange(info.getNumInputs())) {
        info.setInput(i, DimsOrder::fromNumDims(info.getInput(i).numDims()));
    }

    for (auto i : irange(info.getNumOutputs())) {
        info.setOutput(i, DimsOrder::fromNumDims(info.getOutput(i).numDims()));
    }
}

void vpux::IE::fillDefaultLayoutInfo(LayerLayoutInfo& info, FuncRef<bool(size_t)> inputFilter,
                                     FuncRef<bool(size_t)> outputFilter) {
    for (auto i : irange(info.getNumInputs()) | filtered(inputFilter)) {
        info.setInput(i, DimsOrder::fromNumDims(info.getInput(i).numDims()));
    }

    for (auto i : irange(info.getNumOutputs()) | filtered(outputFilter)) {
        info.setOutput(i, DimsOrder::fromNumDims(info.getOutput(i).numDims()));
    }
}

//
// EltwiseOp
//

mlir::LogicalResult vpux::IE::verifyEltwiseOp(mlir::Operation* op) {
    if (!mlir::isa<IE::LayerOpInterface>(op)) {
        return errorAt(op, "EltwiseOp trait is applied to non layer operation");
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation with multiple results can't be EltwiseOp");
    }

    const auto outputShape = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
    if (llvm::none_of(op->getOperands(), [&](mlir::Value operand) {
            return operand.getType().cast<vpux::NDTypeInterface>().getShape() == outputShape;
        })) {
        return errorAt(op, "EltwiseOp must have at least one input shape equal to the output shape");
    }

    return mlir::success();
}

vpux::IE::LayerDataInfo<mlir::Type> vpux::IE::getElemTypeInfo(mlir::Operation* op) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputTypes.push_back(val.getType().cast<vpux::NDTypeInterface>().getElementType());
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputTypes.push_back(val.getType().cast<vpux::NDTypeInterface>().getElementType());
    }

    return vpux::IE::LayerDataInfo<mlir::Type>(std::move(inputTypes), std::move(outputTypes));
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.cpp.inc>

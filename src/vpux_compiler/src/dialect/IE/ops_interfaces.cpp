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

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>

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

bool vpux::IE::isCompatibleTensorTypes(mlir::TypeRange lhs, mlir::TypeRange rhs,
                                       IE::TypeComparisonMode elemComparisonModes, bool checkDimsOrder,
                                       bool checkMemSpace, bool checkSparsity) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto p : zip(lhs, rhs)) {
        const auto lhsType = std::get<0>(p).dyn_cast<mlir::RankedTensorType>();
        const auto rhsType = std::get<1>(p).dyn_cast<mlir::RankedTensorType>();

        if (lhsType == nullptr || rhsType == nullptr) {
            return false;
        }

        if (checkSparsity) {
            if (IE::isSparse(lhsType) != IE::isSparse(rhsType)) {
                return false;
            }
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
            const auto order1 = IE::getOrder(lhsType);
            const auto order2 = IE::getOrder(rhsType);

            if (order1 != order2) {
                return false;
            }
        }

        if (checkMemSpace) {
            const auto memSpace1 = IE::getMemorySpace(lhsType);
            const auto memSpace2 = IE::getMemorySpace(rhsType);

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
// TilingBuilderOpInterface
//

mlir::Value vpux::IE::makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal,
                               const TileInfo& tile, StringRef valName) {
    if (tile.shape == getShape(origVal)) {
        return origVal;
    }

    const auto tileName = llvm::formatv("{0} tile {1}", valName, tile.offsets).str();
    const auto loc = appendLoc(baseLoc, tileName);

    auto sliceOp = builder.create<IE::SliceOp>(loc, origVal, tile.offsets, tile.shape);
    return sliceOp.result();
}

//
// TilingInfoOpInterface
//

mlir::LogicalResult vpux::IE::verifyTilingInfo(mlir::Operation* op) {
    if (!mlir::isa<IE::TilingBuilderOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' provides TilingInfoOpInterface, but not TilingBuilderOpInterface",
                       op->getName());
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Unsupported operation '{0}', it must have one and only one result", op->getName());
    }

    return mlir::success();
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

    return mlir::success();
}

SmallVector<int64_t> vpux::IE::getMaxNumTiles(mlir::Operation* op) {
    const auto& outputShape = getShape(op->getResult(0));

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());

    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getOutputChannelAlignment();
    }

    const auto maxChannelTiles = outputShape[Dims4D::Act::C] / minChannelSize;

    SmallVector<int64_t> maxNumTiles(outputShape.begin(), outputShape.end());
    maxNumTiles[Dims4D::Act::C.ind()] = maxChannelTiles;

    return maxNumTiles;
}

InputTiling vpux::IE::backInferEltwiseTile(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    SmallVector<TileInfo> inputTiles;
    for (auto& origInput : op->getOpOperands()) {
        const auto curShape = getShape(origInput.get());
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile eltwise operation '{0}' at '{1}', which has operands with different rank",
                          op->getName(), op->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }

    return TilingInfo{inputTiles};
}

vpux::IE::LayerDataInfo<mlir::Type> vpux::IE::getElemTypeInfo(mlir::Operation* op) {
    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(op->getNumOperands());
    for (const auto& val : op->getOperands()) {
        inputTypes.push_back(val.getType().cast<mlir::MemRefType>().getElementType());
    }

    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(op->getNumResults());
    for (const auto& val : op->getResults()) {
        outputTypes.push_back(val.getType().cast<mlir::MemRefType>().getElementType());
    }

    return vpux::IE::LayerDataInfo<mlir::Type>(std::move(inputTypes), std::move(outputTypes));
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.cpp.inc>

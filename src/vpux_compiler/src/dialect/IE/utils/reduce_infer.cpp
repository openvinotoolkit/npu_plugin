//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
constexpr vpux::StringLiteral keep_dims = "keep_dims";

mlir::LogicalResult vpux::IE::inferReduceReturnTypeComponents(
        mlir::Location loc, mlir::Value input, bool keepDims, SmallVector<int64_t>& axes,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto inType = input.getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axes.begin(), axes.end()) == axes.end();
    if (!isAllUnique) {
        return errorAt(loc, "Axes values should be unique");
    }

    // Add to outShape the values with indices not found in axes_set.
    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inShape.size(); i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            outShape.push_back(inShape[i]);
        } else if (keepDims) {
            outShape.push_back(1);
        }
    }

    if (outShape.size() == 0) {
        outShape.push_back(1);
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

// This function calculate outputDimOrder in case some axes are reduced.
// Example:
// NCHW 1234, remove N 1 -> 234 -> 123 (CHW)
//            remove H 2 -> 134 -> 123 (CHW)
//            remove W 3 -> 124 -> 123 (CHW)
//            remove W 4 -> 123 -> 123 (CHW)
//
// NHWC 1342, remove N 1 -> 342 -> 231 (HWC)
//            remove H 3 -> 142 -> 132 (CWH)
//            remove W 4 -> 132 -> 132 (CWH)
//            remove C 2 -> 134 -> 123 (CHW)
vpux::DimsOrder vpux::IE::calculateReducedOutputLayout(vpux::DimsOrder inputDimOrder, mlir::SmallVector<int64_t> axes) {
    auto inputCodeOrder = inputDimOrder.code();
    vpux::DimsOrder::StorageType outputCodeOrder = 0;
    uint64_t multiply = 0;
    while (inputCodeOrder) {
        auto it = std::find(axes.begin(), axes.end(), inputCodeOrder % (1 << vpux::DimsOrder::BITS_PER_DIM));
        if (it == axes.end()) {
            int64_t numberToInsert = inputCodeOrder % (1 << vpux::DimsOrder::BITS_PER_DIM);
            auto numSmallerReducedAxes = 0;
            for (auto axis : axes) {
                if (axis < numberToInsert) {
                    numSmallerReducedAxes++;
                }
            }
            numberToInsert -= numSmallerReducedAxes;
            outputCodeOrder +=
                    static_cast<uint64_t>(numberToInsert) * (1ULL << (vpux::DimsOrder::BITS_PER_DIM * multiply));
            multiply++;
        }
        inputCodeOrder = inputCodeOrder / (1 << vpux::DimsOrder::BITS_PER_DIM);
    }
    return vpux::DimsOrder::fromCode(outputCodeOrder);
}

void vpux::IE::inferReduceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    const bool keepDims = op->hasAttr(keep_dims) ? op->getAttr(keep_dims) != nullptr : false;
    llvm::SmallVector<int64_t> axesVec;
    if (op->getNumOperands() > 1) {
        axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(op->getOperand(1)));
    }

    const auto filter = [](size_t ind) {
        return ind != 0;
    };

    IE::fillDefaultLayoutInfo(info, filter, filter);

    const auto mainOrder = info.getInput(0);

    if (!keepDims) {
        info.setOutput(0, calculateReducedOutputLayout(mainOrder, axesVec));
    } else {
        info.setOutput(0, mainOrder);
    }
}

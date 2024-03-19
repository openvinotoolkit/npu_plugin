//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"

namespace vpux {
namespace IE {

mlir::ArrayAttr getNewConcatOffsetsParameters(mlir::ArrayAttr oldOffsets, mlir::ArrayAttr dimsMappingAttr,
                                              mlir::OperandRange oldInputs, ArrayRef<vpux::ShapeRef> newInputShapes,
                                              ShapeRef reshapeShape, mlir::DenseSet<int64_t> modifiedAxes) {
    const auto oldOffsetsList = parseIntArrayOfArrayAttr<int64_t>(oldOffsets);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimsMappingAttr);

    SmallVector<SmallVector<int64_t>> newOffsetsList;
    newOffsetsList.reserve(oldOffsetsList.size());

    for (auto inputIndex : irange(oldOffsetsList.size())) {
        const auto oldInputShape = getShape(oldInputs[inputIndex]).raw();
        const auto newInputShape = newInputShapes[inputIndex].raw();
        const auto oldOffset = oldOffsetsList[inputIndex];

        SmallVector<int64_t> newOffset(newInputShape.size(), 0);
        for (const auto oldConcatDim : modifiedAxes) {
            for (const auto& dim : dimMapping[oldConcatDim]) {
                // Condition "reshapeShape[Dim(dim)] != 1" is added to handle the following case:
                // Concat on a dimension and then unsqueeze that dimension, e.g.:
                // 2 x ([1, 3]) -> Concat -> ([2, 3]) -> AffineReshape: dimMapping={[0, 1, 2], [3]} -> ([1, 2, 1, 3])
                if (oldInputShape[oldConcatDim] == newInputShape[dim] && reshapeShape[Dim(dim)] != 1) {
                    newOffset[dim] = oldOffset[oldConcatDim];
                    break;
                }
            }
        }

        newOffsetsList.push_back(newOffset);
    }

    // Make sure that there is at least one offset is set
    bool isOffsetSet = std::any_of(newOffsetsList.begin(), newOffsetsList.end(), [](ArrayRef<int64_t> v) {
        return std::any_of(v.begin(), v.end(), [](int64_t i) {
            return i != 0;
        });
    });
    VPUX_THROW_UNLESS(isOffsetSet == true, "No valid concat offset was found during ConcatReshapeConcat rewritten");

    return getIntArrayOfArray(dimsMappingAttr.getContext(), ArrayRef(newOffsetsList));
}

mlir::DenseSet<int64_t> getConcatModifiedAxis(IE::ConcatOp origOp) {
    mlir::DenseSet<int64_t> modifiedAxes;
    const auto offsets = parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsetsAttr());

    for (size_t i = 0; i < offsets.size(); i++) {
        for (size_t j = 0; j < offsets[i].size(); ++j) {
            if (offsets[i][j] != 0) {
                modifiedAxes.insert(j);
            }
        }
    }

    return modifiedAxes;
}

SmallVector<int64_t> calculateInputShapeAfterSwitchConcatAndAffineReshape(mlir::Value input, IE::ConcatOp concatOp,
                                                                          IE::AffineReshapeOp reshapeOp) {
    const auto affineOutShape = getShape(reshapeOp.getOutput());
    const auto modifiedAxes = getConcatModifiedAxis(concatOp);

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());
    SmallVector<int64_t> newShapeVec(affineOutShape.size());
    for (size_t dimIdx = 0; dimIdx < affineOutShape.size(); dimIdx++) {
        auto axisIt = llvm::find_if(modifiedAxes, [&](int64_t modifiedAxis) {
            for (auto& mappedIdx : dimMapping[modifiedAxis]) {
                if (affineOutShape[Dim(mappedIdx)] == 1) {
                    continue;
                } else {
                    return dimIdx == checked_cast<size_t>(mappedIdx);
                }
            }
            return false;
        });
        if (axisIt != modifiedAxes.end() && affineOutShape[Dim(dimIdx)] != 1) {
            newShapeVec[dimIdx] = getShape(input)[Dim(*axisIt)];
        } else if (affineOutShape[Dim(dimIdx)] == 1) {
            newShapeVec[dimIdx] = 1;
        } else {
            newShapeVec[dimIdx] = affineOutShape[Dim(dimIdx)];
        }
    }
    return newShapeVec;
}

}  // namespace IE
}  // namespace vpux

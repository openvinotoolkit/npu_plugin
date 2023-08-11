//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"

namespace {
struct MinDimension {
    std::size_t& shapeIdx;
    llvm::ArrayRef<int64_t> shape;
    int64_t largeDimQuotient;

    MinDimension(std::size_t& shapeIdx, llvm::ArrayRef<int64_t> shape, const int64_t largeDimQuotient)
            : shapeIdx(shapeIdx), shape(shape), largeDimQuotient(largeDimQuotient){};
};
}  // namespace

namespace vpux {
namespace IE {

void handleConsecutiveOnes(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape, std::size_t& startIn,
                           std::size_t& startOut, SmallVector<SmallVector<int64_t>>& reassociationVec) {
    std::size_t endIn = startIn;
    while (endIn < inShape.size() && inShape[endIn] == 1) {
        endIn++;
    }

    std::size_t endOut = startOut;
    while (endOut < outShape.size() && outShape[endOut] == 1) {
        endOut++;
    }

    for (; startIn < endIn && startOut < endOut; ++startIn, ++startOut) {
        reassociationVec[startIn].push_back(static_cast<int64_t>(startOut));
    }

    while (startIn < endIn) {
        reassociationVec[startIn].push_back(static_cast<int64_t>(startOut - 1));
        startIn++;
    }

    while (startOut < endOut) {
        reassociationVec[startIn - 1].push_back(static_cast<int64_t>(startOut));
        startOut++;
    }
}

// Note: When having dims equal to 1 in one of the shapes that do not have a corresponding 1 in the other shape, there
// might be multiple dim associations possible. The current algorithm takes only one into consideration.
// E.g.: 1 x 2 x 2 x 1 x 2 x 3 -> 1 x 4 x 6 has 2 possible mappings:
//      {0} -> {0}, {1, 2, 3} -> {1}, {4, 5} -> {2} (this one is computed by the fcn below)
//      {0} -> {0}, {1, 2} -> {1}, {3, 4, 5} -> {2}
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMap(ArrayRef<int64_t> inShape,
                                                                       ArrayRef<int64_t> outShape) {
    const auto inSize = inShape.size();
    const auto outSize = outShape.size();

    const auto nextDimIsOne = [](ArrayRef<int64_t> shape, const std::size_t index) -> bool {
        return index + 1 < shape.size() && shape[index + 1] == 1;
    };

    SmallVector<SmallVector<int64_t>> reassociationVec(inSize);
    std::size_t inIdx = 0, outIdx = 0;
    for (; inIdx < inSize && outIdx < outSize; ++inIdx, ++outIdx) {
        if (inShape[inIdx] == 1 && outShape[outIdx] == 1) {
            // Pair dims equal to 1 that have corresponding dims in the other shape
            handleConsecutiveOnes(inShape, outShape, inIdx, outIdx, reassociationVec);

            if (inIdx >= inSize || outIdx >= outSize) {
                break;
            }
        }

        // If both dims are equal, pick the one that has a dim of 1 after it. If there is no corresponding dim equal to
        // 1 in the other shape, the mapping dim_large = 1 x dim_small will be added. Without that extra condition,
        // there could be cases where that extra 1 remains floating, leading the algorithm to decide that there is no
        // valid mapping between shapes.
        const bool isInputSmallerDim = inShape[inIdx] < outShape[outIdx] ||
                                       (inShape[inIdx] == outShape[outIdx] && nextDimIsOne(inShape, inIdx));
        auto minimum = isInputSmallerDim ? MinDimension(inIdx, inShape, outShape[outIdx])
                                         : MinDimension(outIdx, outShape, inShape[inIdx]);

        do {
            if (minimum.largeDimQuotient % minimum.shape[minimum.shapeIdx] != 0)
                return mlir::failure();

            reassociationVec[inIdx].push_back(static_cast<int64_t>(outIdx));

            minimum.largeDimQuotient /= minimum.shape[minimum.shapeIdx];

            if (minimum.largeDimQuotient == 1) {
                // Exit loop if the next dim isn't 1 or if there are 1s on next dim of both shapes
                if (!nextDimIsOne(minimum.shape, minimum.shapeIdx) ||
                    (nextDimIsOne(inShape, inIdx) && nextDimIsOne(outShape, outIdx))) {
                    break;
                }
            }

            ++minimum.shapeIdx;
        } while (minimum.shapeIdx < minimum.shape.size());
    }

    // One of the shapes has trailing 1s that cannot be the result of decomposing the last dim of the other shape
    if (inIdx < inSize || outIdx < outSize) {
        return mlir::failure();
    }

    return reassociationVec;
}

}  // namespace IE
}  // namespace vpux

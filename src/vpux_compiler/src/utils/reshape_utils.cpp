//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/reshape_utils.hpp"

#include <numeric>

using namespace vpux;

SmallVector<MemDimArr> getOutMemDimsCandidates(MemShapeRef inMemShape, MemShapeRef outMemShape, MemDim inMemDim) {
    const size_t targetDimSize = checked_cast<size_t>(inMemShape[inMemDim]);
    SmallVector<MemDimArr> outMemDims;
    // For Example: inMemShape: 1x512x512x16, outMemShape: 16x32x512x16, inMemDim: 'H'
    // The 'targetDimSize' will have two candidates: [[0, 1],[2]]
    // Candidate 1: [0, 1] input 'H' split into output 'N' and 'C'
    // Candidate 2: [2] input 'H' split into output 'H'
    for (size_t dimIdx = 0; dimIdx < outMemShape.size(); dimIdx++) {
        size_t accumulateSize = 1;
        size_t beginIdx = dimIdx;
        MemDimArr currMemDims;
        while (beginIdx < outMemShape.size()) {
            accumulateSize = accumulateSize * outMemShape[MemDim(beginIdx)];
            currMemDims.push_back(MemDim(beginIdx));
            if (accumulateSize == targetDimSize) {
                outMemDims.push_back(currMemDims);
            } else if (accumulateSize > targetDimSize) {
                break;
            }
            beginIdx++;
        }
    }
    return outMemDims;
}

std::optional<MemDimArr> vpux::deduceLegalOutputMemDims(MemShapeRef inMemShape, MemShapeRef outMemShape,
                                                        MemDim inMemDim) {
    const auto outMemDimsCandidates = getOutMemDimsCandidates(inMemShape, outMemShape, inMemDim);
    if (outMemDimsCandidates.empty()) {
        return std::nullopt;
    }

    auto getAccumulateSize = [](MemShapeRef memShape, auto beginIdx, auto endIdx) {
        VPUX_THROW_UNLESS(checked_cast<int32_t>(beginIdx) <= checked_cast<int32_t>(endIdx) &&
                                  memShape.begin() + endIdx <= memShape.end(),
                          "Got unexpect memShape");
        return std::accumulate(memShape.begin() + beginIdx, memShape.begin() + endIdx, int64_t(1),
                               std::multiplies<int64_t>());
    };

    // For Example: inMemShape: 1x512x512x16, outMemShape: 16x32x512x16, inMemDim: H
    // The 'outMemDims' will have two candidates: [[0, 1],[2]]
    // Candidate 1: outMemDims is [0, 1]
    // inTotalLeftSize(1x512) != outTotalLeftSize(1) && inTotalRightSize(16) != outTotalRightSize(512x16)
    // Candidate 2: outMemDims is [2]
    // inTotalLeftSize(1x512) == outTotalLeftSize(16x32) && inTotalRightSize(16) == outTotalRightSize(16)
    // The candidate 2 is legal candidate
    for (auto& outMemDims : outMemDimsCandidates) {
        const auto inTotalLeftShapeSize = getAccumulateSize(inMemShape, 0, inMemDim.ind());
        const auto inTotalRightShapeSize = getAccumulateSize(inMemShape, inMemDim.ind() + 1, inMemShape.size());
        const auto outTotalLeftShapeSize = getAccumulateSize(outMemShape, 0, outMemDims.front().ind());
        const auto outTotalRightShapeSize =
                getAccumulateSize(outMemShape, outMemDims.back().ind() + 1, outMemShape.size());
        if (inTotalLeftShapeSize == outTotalLeftShapeSize && inTotalRightShapeSize == outTotalRightShapeSize) {
            return outMemDims;
        }
    }
    return std::nullopt;
}

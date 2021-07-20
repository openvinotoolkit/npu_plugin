//
// Copyright 2021 Intel Corporation.
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

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>

namespace vpux {
namespace edsl {

struct AutoTileParams {
    uint64_t totalBuffer;
    uint32_t minCount;
    uint32_t maxCount;
    uint64_t minInnerBuffer;
    uint32_t cacheWidth;
    uint32_t vectorWidth;
    llvm::StringRef processingTags;
    llvm::StringRef outerTags;
    llvm::StringRef innerTags;
    bool outputIndicesOnly;
    bool accIndicesOnly;
    bool noNegativeIndex;
};

llvm::SmallVector<int64_t, 8> computeBestTile(mlir::AffineParallelOp parallelOp, const AutoTileParams& params);

}  // namespace edsl
}  // namespace vpux

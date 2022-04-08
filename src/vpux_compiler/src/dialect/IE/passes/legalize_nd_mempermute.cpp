//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <map>

using namespace vpux;

namespace {

//
// LegalizeNDMemPermutePass
//

class LegalizeNDMemPermutePass final : public IE::LegalizeNDMemPermuteBase<LegalizeNDMemPermutePass> {
public:
    explicit LegalizeNDMemPermutePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Normalize permutation vector
// Example: [1, 3, 7, 6] -> [0, 1, 3, 2]

void normalizePermutation(SmallVector<uint32_t>& vec) {
    SmallVector<uint32_t> sorted(vec);
    llvm::DenseMap<uint32_t, uint32_t> helper;
    std::sort(sorted.begin(), sorted.end());

    for (size_t i = 0; i < sorted.size(); ++i) {
        helper.insert(std::make_pair(sorted[i], static_cast<uint32_t>(i)));
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = helper[vec[i]];
    }
}

//
// safeRunOnFunc
//
void LegalizeNDMemPermutePass::safeRunOnFunc() {
    // Only enabled for VPUX37XX where Tiling for SW kernels is limited to 4D ops. Not all ops can be
    // reduced to 4D! As a future task these ops they could be decomposed into multiple MemPermutes that are 4D.
    // ToDo: Remove pass after limitation.
    auto func = getFunction();

    func.walk([this](IE::MemPermuteOp origOp) {
        auto _log = this->_log;
        auto ctx = &this->getContext();

        auto inputType = origOp.input().getType().cast<NDTypeInterface>();

        if (inputType.getRank() == 4) {
            return;
        }

        auto outputType = origOp.output().getType().cast<NDTypeInterface>();
        auto permutation = origOp.mem_perm();

        _log.trace("Got MemPermute op at '{0}' with shape '{1}' and permutation map '{2}'", origOp->getLoc(),
                   inputType.getShape(), permutation);

        auto memShape = to_small_vector(inputType.getMemShape());
        auto dstOrder = origOp.dst_order();
        auto origPermVec = DimsOrder::fromAffineMap(permutation).toPermutation();

        // Example of origPermVec
        // origPermVec = [d0, d1, d2, d3] -> [d1, d2, d3, d0];
        // origPermVec[0] = 1, origPermVec[1] = 2, origPermVec[2] = 3, origPermVec[3] = 0
        SmallVector<uint32_t> permVec;
        SmallVector<int64_t> shapeVec;

        for (auto d : origPermVec) {
            // Dims with size 1 are dropped
            if (memShape[d.ind()] != 1) {
                permVec.push_back(static_cast<uint32_t>(d.ind()));
            }
        }

        for (auto dimSize : memShape) {
            // Dims with size 1 are dropped
            if (dimSize != 1) {
                shapeVec.push_back(dimSize);
            }
        }

        normalizePermutation(permVec);

        // Merge dims that are adjacent before and after permutation
        // Example:
        // memShape =[2, 4, 25, 255, 255]
        // permVec = [d0, d1, d2, d3, d4] -> [d0, d4, d1, d2, d3]
        //
        // mergedShape = [2, 25500, 255]
        // mergedPermutation = [d0, d1, d2] -> [d0, d2, d1]
        SmallVector<uint32_t> mergedPermutation;
        SmallVector<int64_t> mergedShape;

        std::map<uint32_t, int64_t> mergedShapeMap;
        size_t j = 0;
        for (size_t i = 0; i < static_cast<size_t>(permVec.size()); i = j) {
            int64_t dimSize = shapeVec[permVec[i]];
            for (j = i + 1; j < static_cast<size_t>(permVec.size()) && (permVec[j - 1] + 1 == permVec[j]); ++j) {
                dimSize *= shapeVec[permVec[j]];
            }

            mergedShapeMap.insert(std::make_pair(permVec[i], dimSize));

            mergedPermutation.push_back(permVec[i]);
        }

        if (mergedPermutation.size() > 4) {
            _log.trace("Could only reduce to rank '{0}'. Pattern is not applied", mergedPermutation.size());
            return;
        }

        // Keys iterated in ascending order
        for (auto p : mergedShapeMap) {
            mergedShape.push_back(p.second);
        }

        // Normalize vectors
        normalizePermutation(mergedPermutation);

        // Pad to 4D if needed (Tiling hardcoded to 4D)
        for (size_t i = mergedPermutation.size(); i < 4; i++) {
            mergedPermutation.push_back(static_cast<uint32_t>(i));
            mergedShape.push_back(1);
        }

        auto reducedPermutation = mlir::AffineMap::getPermutationMap(makeArrayRef(mergedPermutation), ctx);

        mlir::OpBuilder builder(origOp.getOperation());

        // Cast to canonical order for convenience
        auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(static_cast<uint32_t>(inputType.getRank()), ctx);
        auto inputCast = builder.create<IE::PermuteCastOp>(origOp.getLoc(), origOp.input(), identityMap, identityMap);

        // Build input reshape operation
        auto reducedShapeAttr = getIntArrayAttr(ctx, mergedShape);
        auto inputReshape = builder.create<IE::ReshapeOp>(origOp.getLoc(), inputCast.output(), /*shape=*/nullptr,
                                                          /*special_zero=*/nullptr, reducedShapeAttr);

        // Build reduced permutation operation
        auto newPermute = builder.create<IE::MemPermuteOp>(
                origOp.getLoc(), inputReshape.output(),
                mlir::AffineMap::getMultiDimIdentityMap(static_cast<uint32_t>(mergedShape.size()), ctx),
                reducedPermutation);

        // Reshape to original output shape
        auto outputShape = outputType.getMemShape();
        auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
        auto outputReshape = builder.create<IE::ReshapeOp>(origOp.getLoc(), newPermute.output(), /*shape=*/nullptr,
                                                           /*special_zero=*/nullptr, outputShapeAttr);

        // Set destination order
        auto newSequence = builder.create<IE::PermuteCastOp>(
                origOp.getLoc(), outputReshape.output(), dstOrder,
                mlir::AffineMap::getMultiDimIdentityMap(static_cast<uint32_t>(outputShape.size()), ctx));

        origOp.replaceAllUsesWith(newSequence.output());
        origOp.erase();

        _log.nest().trace("Replaced with shape '{0}' and permutation map '{1}'", mergedShape, reducedPermutation);
    });
}

}  // namespace

//
// createConvertToMemPermutePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeNDMemPermutePass(Logger log) {
    return std::make_unique<LegalizeNDMemPermutePass>(log);
}

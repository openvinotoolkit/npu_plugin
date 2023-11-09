//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes/optimize_slice_expand.hpp"
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/range.hpp"

#include <numeric>

using namespace vpux;

//
// OptimizeSliceImplicitExpand
//

mlir::LogicalResult vpux::IE::genericOptimizeSliceImplicitExpand(IE::ExpandOp layerOp, mlir::Operation* implicitOp,
                                                                 mlir::PatternRewriter& rewriter) {
    // avoid unsupported cases
    if (implicitOp == nullptr || implicitOp->getNumResults() != 1 || !implicitOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto inputs = implicitOp->getOperands();
    auto expandedShape = to_small_vector(getShape(layerOp.output()));
    auto implicitShape = to_small_vector(getShape(implicitOp->getResult(0)));

    if (llvm::any_of(inputs, [&](mlir::Value operand) {
            auto sliceOp = operand.getDefiningOp<IE::SliceOp>();
            return sliceOp == nullptr ||
                   getShape(sliceOp.source())[Dims4D::Act::C] < expandedShape[Dims4D::Act::C.ind()];
        })) {
        return mlir::failure();
    }

    // avoid situation with Concat over C to be expanded incorrectly
    if (inputs.size() > 1) {
        const auto channelAxis = std::accumulate(inputs.begin(), inputs.end(), (int64_t)0,
                                                 [&](int64_t accumulator, mlir::Value operand) {
                                                     return accumulator + getShape(operand)[Dims4D::Act::C];
                                                 });

        if (channelAxis == implicitShape[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }
    }

    // Expand cannot be executed on N/H/W axis
    for (auto index : irange(expandedShape.size())) {
        if (index == checked_cast<uint32_t>(Dims4D::Act::C.ind()) || expandedShape[index] == implicitShape[index]) {
            continue;
        }

        return mlir::failure();
    }

    // check here if operation changes N/H/W axis, Slice/Expand should not change this as well
    auto outputShape = to_small_vector(getShape(implicitOp->getResult(0)));
    for (mlir::Value input : inputs) {
        auto sliceOp = input.getDefiningOp<IE::SliceOp>();
        auto inputShape = to_small_vector(getShape(input));

        if (inputShape.size() != outputShape.size()) {
            return mlir::failure();
        }

        for (auto index : irange(inputShape.size())) {
            if (index == checked_cast<uint32_t>(Dims4D::Act::C.ind())) {
                continue;
            }

            // if operation changes axis, check if Slice doesn't change this axis as well
            if (getShape(sliceOp.source())[Dim(index)] != getShape(sliceOp.result())[Dim(index)]) {
                return mlir::failure();
            }
        }
    }

    rewriter.startRootUpdate(implicitOp);
    rewriter.setInsertionPoint(implicitOp);

    for (auto i : irange<size_t>(0, inputs.size())) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(inputs[i].getDefiningOp());

        auto parentShape = to_small_vector(getShape(sliceOp.source()));

        mlir::Value newInput = sliceOp.source();
        if (parentShape[Dims4D::Act::C.ind()] != expandedShape[Dims4D::Act::C.ind()]) {
            auto sliceSizes = parseIntArrayAttr<int64_t>(sliceOp.static_sizesAttr());
            sliceSizes[Dims4D::Act::C.ind()] = expandedShape[Dims4D::Act::C.ind()];
            newInput = rewriter.create<IE::SliceOp>(sliceOp.getLoc(), newInput, sliceOp.static_offsetsAttr(),
                                                    getIntArrayAttr(layerOp.getContext(), makeArrayRef(sliceSizes)));
        }

        implicitOp->getOpOperand(static_cast<uint32_t>(i)).set(newInput);
    }

    for (mlir::Value result : implicitOp->getResults()) {
        auto resultType = result.getType().cast<vpux::NDTypeInterface>();
        auto resultShape = to_small_vector(getShape(result));
        resultShape[Dims4D::Act::C.ind()] = expandedShape[Dims4D::Act::C.ind()];

        const auto outType = resultType.changeShape(ShapeRef(resultShape));

        result.setType(outType);
    }

    for (auto* user : llvm::make_early_inc_range(implicitOp->getUsers())) {
        rewriter.replaceOp(user, implicitOp->getResults());
    }

    rewriter.finalizeRootUpdate(implicitOp);

    return mlir::success();
}

//
// Single slice beneficial pattern:
//
//               input1
//                 |
//      input0    Slice                   input0    input 1
//          \      /                          \      /
//           Concat            ==>             Concat
//              |                                 |
//           Expand                             output
//              |
//            output
//
bool isSingleSliceBeneficialPattern(IE::ConcatOp concatOp, IE::ExpandOp layerOp) {
    const auto inputs = concatOp.getInputs();
    auto expandedShape = to_small_vector(getShape(layerOp.output()));
    auto concatShape = to_small_vector(getShape(concatOp.output()));
    uint32_t expandAxisNum = 0;
    SmallVector<uint32_t> expandAxisSet;
    uint32_t inputSliceNum = 0;
    mlir::DenseSet<IE::SliceOp> inputSliceOps{};

    // Infer sliceOp number and sliceOp
    for (mlir::Value input : inputs) {
        if (auto sliceOp = input.getDefiningOp<IE::SliceOp>()) {
            inputSliceNum++;
            if (inputSliceNum == 1) {
                inputSliceOps.insert(sliceOp);
            }
        }
    }

    // Infer expandOp axis
    for (auto index : irange(expandedShape.size())) {
        if (expandedShape[index] != concatShape[index]) {
            expandAxisNum++;
            expandAxisSet.push_back(static_cast<uint32_t>(index));
        }
    }

    if (inputs.size() <= 1 || inputSliceNum != 1 || expandAxisNum != 1) {
        return false;
    }

    auto inputSliceOp = *inputSliceOps.begin();
    const auto expandAxis = expandAxisSet.front();

    // Expand axis equals to Slice axis and Concat axis
    auto sliceOutShape = to_small_vector(getShape(inputSliceOp->getResult(0)));
    auto sliceInShape = to_small_vector(getShape(inputSliceOp.source()));
    for (auto index : irange(sliceOutShape.size())) {
        if (index != expandAxis &&
            (sliceOutShape[index] != sliceInShape[index] || sliceOutShape[index] != concatShape[index])) {
            return false;
        }
    }

    const auto axisShape = std::accumulate(inputs.begin(), inputs.end(), static_cast<int64_t>(0),
                                           [&](int64_t accumulator, mlir::Value operand) {
                                               return accumulator + getShape(operand)[Dim(expandAxis)];
                                           });
    if (axisShape != concatShape[expandAxis]) {
        return false;
    }

    // Check the slice offset is the last one in Concat
    if (!concatOp.static_offsetsAttr()) {
        return false;
    }
    const auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.static_offsetsAttr());
    uint32_t lastOffset = 0;
    for (const auto& p : zip(concatOp.inputs(), concatOffsets)) {
        auto curInput = std::get<0>(p);
        auto curSlice = curInput.getDefiningOp<IE::SliceOp>();
        if (curSlice != nullptr) {
            const auto curOffset = std::get<1>(p);
            const auto curOffsetShape = Shape(curOffset);
            lastOffset = curOffsetShape[Dim(expandAxis)];
        }
    }

    return (lastOffset + sliceOutShape[expandAxis]) == concatShape[expandAxis] &&
           (lastOffset + sliceInShape[expandAxis]) == expandedShape[expandAxis];
}

//
// OptimizeSingleSliceConcatExpand
//

mlir::LogicalResult vpux::IE::OptimizeSingleSliceConcatExpand::matchAndRewrite(IE::ExpandOp layerOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    auto concatOp = layerOp.input().getDefiningOp<IE::ConcatOp>();
    // avoid unsupported cases
    if (concatOp == nullptr || concatOp->getNumResults() != 1 || !concatOp->hasOneUse()) {
        return mlir::failure();
    }

    auto expandedShape = to_small_vector(getShape(layerOp.output()));
    auto concatShape = to_small_vector(getShape(concatOp.output()));

    bool sameAxisSliceConcatExpand = isSingleSliceBeneficialPattern(concatOp, layerOp);
    if (!sameAxisSliceConcatExpand) {
        return mlir::failure();
    }

    SmallVector<uint32_t> expandAxis;
    for (auto index : irange(expandedShape.size())) {
        if (expandedShape[index] != concatShape[index]) {
            expandAxis.push_back(static_cast<uint32_t>(index));
        }
    }

    const auto inputs = concatOp.getInputs();
    mlir::Operation* updateOp = concatOp.getOperation();

    rewriter.startRootUpdate(updateOp);
    rewriter.setInsertionPoint(updateOp);

    for (auto i : irange<size_t>(0, inputs.size())) {
        auto sliceOp = inputs[i].getDefiningOp<IE::SliceOp>();
        if (sliceOp == nullptr) {
            continue;
        }
        auto newInput = sliceOp.source();
        updateOp->getOpOperand(static_cast<uint32_t>(i)).set(newInput);
    }

    const auto expandAxisData = expandAxis.front();
    for (mlir::Value result : updateOp->getResults()) {
        auto resultType = result.getType().cast<vpux::NDTypeInterface>();
        auto resultShape = to_small_vector(getShape(result));
        resultShape[expandAxisData] = expandedShape[expandAxisData];
        const auto outType = resultType.changeShape(ShapeRef(resultShape));
        result.setType(outType);
    }

    for (auto* user : llvm::make_early_inc_range(updateOp->getUsers())) {
        rewriter.replaceOp(user, updateOp->getResults());
    }

    rewriter.finalizeRootUpdate(updateOp);

    return mlir::success();
}

//
// OptimizeSliceExpand
//

mlir::LogicalResult vpux::IE::OptimizeSliceExpand::matchAndRewrite(IE::ExpandOp layerOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto sliceOp = layerOp.input().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.static_offsets());
    auto expandedShape = to_small_vector(getShape(layerOp.output()));
    auto parentShape = to_small_vector(getShape(sliceOp.source()));

    bool convertToSlice = true;
    bool convertToExpand = true;

    for (auto index : irange(parentShape.size())) {
        auto newEndOffset = sliceOffset[index] + expandedShape[index];

        convertToSlice &= parentShape[index] >= newEndOffset;
        convertToExpand &= newEndOffset >= parentShape[index] && sliceOffset[index] == 0;
    }

    if (convertToSlice && convertToExpand) {
        return mlir::failure();
    }

    if (convertToExpand) {
        auto newPadsEnd = SmallVector<int64_t>(parentShape.size(), 0);
        for (auto index : irange(parentShape.size())) {
            newPadsEnd[index] = expandedShape[index] - parentShape[index];
        }

        rewriter.replaceOpWithNewOp<IE::ExpandOp>(layerOp, sliceOp.source(), layerOp.pads_beginAttr(),
                                                  getIntArrayAttr(layerOp.getContext(), newPadsEnd));
        return mlir::success();
    }

    if (convertToSlice) {
        rewriter.replaceOpWithNewOp<IE::SliceOp>(layerOp, sliceOp.source(), sliceOp.static_offsetsAttr(),
                                                 getIntArrayAttr(layerOp.getContext(), expandedShape));
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeExpandSlice
//

namespace {
// check if the Expand-Slice pattern could be eliminated or not
// examples that could be eliminated, input shape is [1, 3, 32, 32]
//      Expand pads_begin = [0, 0, 0, 0] pads_end = [0, 13, 0, 0]
//      Slice offset = [0, 0, 0, 0] sizes = [0, 3, 0, 0]
//  or
//      Expand pads_begin = [0, 3, 0, 0] pads_end = [0, 10, 0, 0]
//      Slice offset = [0, 3, 0, 0] sizes = [0, 3, 0, 0]
// examples that could not be eliminated:
//      Expand pads_begin = [0, 3, 0, 0] pads_end = [0, 10, 0, 0]
//      Slice offset = [0, 0, 0, 0] sizes = [0, 3, 0, 0]
//  or
//      Expand pads_begin = [0, 3, 0, 0] pads_end = [0, 10, 0, 0]
//      Slice offset = [0, 0, 0, 0] sizes = [0, 0, 3, 0]
bool expandSliceFusable(IE::ExpandOp expandOp, IE::SliceOp sliceOp) {
    if (expandOp.input().getType() != sliceOp.result().getType()) {
        return false;
    }
    if (sliceOp.source().getDefiningOp() != expandOp.getOperation()) {
        return false;
    }
    auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.pads_begin());
    auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.pads_end());
    auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.static_offsets());
    auto sliceSizes = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
    if (expandPadsBegin.size() != sliceOffsets.size()) {
        return false;
    }
    for (auto index : irange(expandPadsBegin.size())) {
        if ((expandPadsBegin[index] != 0 || sliceOffsets[index] != 0) &&
            sliceOffsets[index] != expandPadsBegin[index]) {
            return false;
        }
    }
    return true;
}
}  // namespace

mlir::LogicalResult vpux::IE::OptimizeExpandSlice::matchAndRewrite(IE::ExpandOp layerOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), layerOp->getName(), layerOp->getLoc());
    auto sliceOp = mlir::dyn_cast<IE::SliceOp>(*layerOp.output().getUsers().begin());

    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    if (layerOp.input().getType() == sliceOp.result().getType()) {
        if (expandSliceFusable(layerOp, sliceOp)) {
            _log.nest().trace("Fuse Expand-Slice pattern");
            sliceOp.result().replaceAllUsesWith(layerOp.input());
            return mlir::success();
        } else {
            return mlir::failure();
        }
    }

    bool convertToSlice = false;
    bool convertToExpand = false;

    auto parentShape = to_small_vector(getShape(layerOp.input()));
    auto slicedShape = to_small_vector(getShape(sliceOp.result()));

    if (slicedShape.size() == parentShape.size()) {
        for (auto index : irange(slicedShape.size())) {
            if (slicedShape[index] < parentShape[index]) {
                convertToSlice = true;
            } else if (slicedShape[index] > parentShape[index]) {
                convertToExpand = true;
            }

            if (convertToSlice && convertToExpand) {
                return mlir::failure();
            }
        }
    } else {
        return mlir::failure();
    }

    if (!convertToSlice && !convertToExpand) {
        return mlir::failure();
    }

    if (convertToExpand) {
        auto padsEnd = parseIntArrayAttr<int64_t>(layerOp.pads_endAttr());

        for (auto index : irange(slicedShape.size())) {
            padsEnd[index] = slicedShape[index] - parentShape[index];
        }

        rewriter.replaceOpWithNewOp<IE::ExpandOp>(sliceOp, layerOp.input(), layerOp.pads_beginAttr(),
                                                  getIntArrayAttr(layerOp.getContext(), padsEnd));
        return mlir::success();
    }
    auto sliceSizes = parseIntArrayAttr<int64_t>(sliceOp.static_sizesAttr());

    for (auto index : irange(slicedShape.size())) {
        sliceSizes[index] = slicedShape[index];
    }

    rewriter.replaceOpWithNewOp<IE::SliceOp>(sliceOp, layerOp.input(), sliceOp.static_offsetsAttr(),
                                             getIntArrayAttr(layerOp.getContext(), sliceSizes));

    return mlir::success();
}

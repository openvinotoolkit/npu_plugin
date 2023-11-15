//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

IE::GroupConvolutionOp createStridedSliceDWConv(mlir::Value input, mlir::ArrayRef<int64_t> strides, mlir::Location loc,
                                                IE::FakeQuantizeOp inputFQ, mlir::PatternRewriter& rewriter,
                                                Logger log) {
    log.nest().trace("Create dw conv {0}: 1x1", loc);
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(
            rewriter, SmallVector<int64_t>{checked_cast<int64_t>(strides[2]), checked_cast<int64_t>(strides[3])});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, 0});
    auto groupAttr = getIntAttr(rewriter, inShape[Dims4D::Act::C]);

    SmallVector<int64_t> outShapeVec(inShape.size());
    for (size_t dimIdx = 0; dimIdx < inShape.size(); dimIdx++) {
        outShapeVec[dimIdx] = (inShape.raw()[dimIdx] - 1) / strides[dimIdx] + 1;
    }
    Shape outShape(outShapeVec);

    const auto elemType = input.getType().cast<vpux::NDTypeInterface>().getElementType();
    auto createConstOp = [&](ShapeRef shape) -> mlir::Value {
        const auto dataStorageType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);

        Const::ContentAttr contentAttr;
        if (elemType.isa<mlir::FloatType>()) {
            float16 weightsValFloat = 1.0f;
            contentAttr = Const::ContentAttr::get(mlir::DenseElementsAttr::get(dataStorageType, weightsValFloat));
        } else {
            int32_t weightsValInt = 1;
            contentAttr = Const::ContentAttr::get(mlir::DenseElementsAttr::get(dataStorageType, weightsValInt));
        }
        contentAttr = contentAttr.broadcast(Dim(0), shape[Dims4D::Act::N]);
        const auto constOutType = mlir::RankedTensorType::get(to_small_vector(shape), elemType);
        auto declOp = rewriter.create<Const::DeclareOp>(loc, constOutType, contentAttr);

        return declOp;
    };

    // OC is equal with IC
    const auto weightShape = Shape{inShape[Dims4D::Act::C], 1, 1, 1};
    auto weights = createConstOp(weightShape);
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(weightShape), elemType);
    // Insert a fake quantize operation after the kernel when necessary.
    if (inputFQ != nullptr) {
        const auto fqArgType = mlir::RankedTensorType::get({}, elemType);

        auto fqLevelsVal = getIntAttr(rewriter, 255);
        auto fqLowVal = VPU::declareFloatConst(rewriter, loc, 0.0f, fqArgType);
        auto fqInHighVal = VPU::declareFloatConst(rewriter, loc, 254.0f, fqArgType);
        auto fqOutHighVal = VPU::declareFloatConst(rewriter, loc, 254.0f, fqArgType);

        auto quantizationForWeights =
                rewriter.create<IE::FakeQuantizeOp>(loc, dataStorageType, weights, fqLowVal, fqInHighVal, fqLowVal,
                                                    fqOutHighVal, fqLevelsVal, inputFQ.auto_broadcastAttr());
        weights = quantizationForWeights.output();
    }

    // IE::ConvolutionOp output type inference sets NCHW output order.
    // Specify convolution output type explicitly.

    const auto origOutType = input.getType().cast<vpux::NDTypeInterface>();
    const auto grpConvOutType = origOutType.changeShape(outShape);

    auto newLoc = appendLoc(loc, "_strided_slice_GroupConv_1_1");
    return rewriter.create<IE::GroupConvolutionOp>(newLoc, grpConvOutType, input, weights, /*bias=*/nullptr,
                                                   stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                   /*post_opAttr=*/nullptr);
}

//
// ConvertStridedSlice2DWConvPass
//

class ConvertStridedSlice2DWConvPass final : public IE::ConvertStridedSlice2DWConvBase<ConvertStridedSlice2DWConvPass> {
public:
    explicit ConvertStridedSlice2DWConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// StridedSliceOpConverter
//

class StridedSliceOpConverter final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    StridedSliceOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::StridedSliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isBenefitialToConvert(IE::StridedSliceOp origOp, ShapeRef newInShape, mlir::ArrayRef<int64_t> strides) const;

private:
    Logger _log;
};

bool StridedSliceOpConverter::isBenefitialToConvert(IE::StridedSliceOp slice, ShapeRef newInShape,
                                                    mlir::ArrayRef<int64_t> strides) const {
    // Check alignment
    const auto stridedSliceInType = slice.input().getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(stridedSliceInType.getElementType());
    auto IW = newInShape[Dims4D::Act::W];
    auto IC = newInShape[Dims4D::Act::C];
    if (IC % alignment == 0) {
        return true;
    }

    // Check stride on W
    const auto strideW = Shape(strides)[Dims4D::Act::W];
    if (strideW > 1) {
        return false;
    }
    // Check if can align
    if (IC % alignment && (IC * IW % alignment) != 0) {
        return false;
    }
    // Check if is output and order is NCHW
    const auto sliceOutput = slice.output();
    const auto user = *sliceOutput.getUsers().begin();
    const auto outOrder = sliceOutput.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    if (user->use_empty() && outOrder == DimsOrder::NCHW) {
        return false;
    }
    return true;
}

mlir::LogicalResult StridedSliceOpConverter::matchAndRewrite(IE::StridedSliceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got IE::StridedSlice Operation '{0}'", origOp->getLoc());

    if (!origOp.begins_attr().hasValue() || !origOp.ends_attr().hasValue() || !origOp.strides_attr().hasValue()) {
        return mlir::failure();
    }

    auto parentAlignIface = origOp.input().getDefiningOp<IE::AlignedChannelsOpInterface>();
    if (parentAlignIface != nullptr) {
        return mlir::failure();
    }

    auto isOne = [](auto val) {
        return val == 1;
    };
    auto strides = parseIntArrayAttr<int64_t>(origOp.strides_attr().getValue());
    if (llvm::all_of(strides, isOne)) {
        _log.trace("If strides on all axis are 1, it is a normal SliceOp");
        return mlir::failure();
    }

    const auto& ctx = origOp.getContext();

    const mlir::Location location = origOp->getLoc();
    const auto inputFQ = origOp.input().getDefiningOp<IE::FakeQuantizeOp>();

    const auto begins = Shape(parseIntArrayAttr<int64_t>(origOp.begins_attr().getValue()));
    const auto inputOffsetsAttr = getIntArrayAttr(ctx, begins);

    const auto input = origOp.input();
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    const auto ends = Shape(parseIntArrayAttr<int64_t>(origOp.ends_attr().getValue()));

    if (inputShape.size() != 4 || strides.size() != 4 || begins.size() != 4 || ends.size() != 4) {
        return mlir::failure();
    }

    auto inOrder = DimsOrder::fromValue(input);
    Shape newInShape = Shape(inputShape.size(), 0);
    for (auto ind : irange(inputShape.size())) {
        auto idx = inOrder.dimAt(ind);
        newInShape[idx] = ends[idx] - begins[idx];
    }

    if (!isBenefitialToConvert(origOp, newInShape, strides)) {
        _log.trace("Cannot or is not benefitial to convert StridedSlice to DWConv");
        return mlir::failure();
    }

    const auto inputShapeAttr = getIntArrayAttr(ctx, to_small_vector(newInShape));
    const auto inputSlice = rewriter.createOrFold<IE::SliceOp>(location, input, inputOffsetsAttr, inputShapeAttr);

    // Create DWConv op
    auto dwConv = createStridedSliceDWConv(inputSlice, strides, location, inputFQ, rewriter, _log);

    _log.trace("Successfully replaced IE::StridedSlice Operation at {0} with IE::GroupConvolution Op",
               origOp->getLoc());
    rewriter.replaceOp(origOp, dwConv->getResult(0));
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertStridedSlice2DWConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<StridedSliceOpConverter>(&ctx, _log);

    auto func = getOperation();

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertStridedSlice2DWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertStridedSlice2DWConvPass(Logger log) {
    return std::make_unique<ConvertStridedSlice2DWConvPass>(log);
}

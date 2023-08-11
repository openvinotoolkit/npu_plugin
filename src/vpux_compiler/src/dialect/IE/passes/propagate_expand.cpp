//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/strides.hpp>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

const int64_t UNEXPANDED_CHANNELS = 3;
const int64_t EXPANDED_CHANNELS = 4;

//
// EltwiseShapeCastRewriter
//

class EltwiseShapeCastRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    EltwiseShapeCastRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("EltwiseShapeCastRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkInput(mlir::Value tensor, int64_t origChannels) const;
    mlir::Value getExpandedInput(mlir::PatternRewriter& rewriter, IE::ShapeCastOp shapeCast, IE::ExpandOp expandOp,
                                 Shape expandedShape) const;

    Logger _log;
};

mlir::Value EltwiseShapeCastRewriter::getExpandedInput(mlir::PatternRewriter& rewriter, IE::ShapeCastOp shapeCast,
                                                       IE::ExpandOp expandOp, Shape expandedShape) const {
    auto tensor = shapeCast.source();
    if (auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(tensor.getDefiningOp())) {
        rewriter.setInsertionPointAfter(permuteQuantize);

        auto pqUsers = permuteQuantize.output().getUsers();
        SmallVector<mlir::Operation*> otherUsers;
        llvm::copy_if(pqUsers, std::back_inserter(otherUsers), [&](auto user) {
            return user != shapeCast.getOperation();
        });

        auto expand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), permuteQuantize.output(), expandOp.pads_begin(),
                                                    expandOp.pads_end());

        if (!otherUsers.empty()) {
            auto sliceOp = rewriter.create<IE::SliceOp>(
                    expandOp.getLoc(), expand.output(), expandOp.pads_begin(),
                    getIntArrayAttr(rewriter.getContext(), getShape(permuteQuantize.output()).raw()));
            for (auto user : otherUsers) {
                user->replaceUsesOfWith(permuteQuantize.output(), sliceOp.result());
            }
        }

        auto newShapeCast = rewriter.create<IE::ShapeCastOp>(
                expandOp.getLoc(), expand.output(), getIntArrayAttr(expandOp.getContext(), expandedShape.raw()));

        return newShapeCast;
    } else if (auto depthToSpace = mlir::dyn_cast_or_null<IE::DepthToSpaceOp>(tensor.getDefiningOp())) {
        rewriter.setInsertionPoint(expandOp);
        auto expand = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), depthToSpace.output(), expandOp.pads_begin(),
                                                    expandOp.pads_end());

        return rewriter.create<IE::ShapeCastOp>(expandOp.getLoc(), expand.output(),
                                                getIntArrayAttr(expandOp.getContext(), expandedShape.raw()));
    }

    VPUX_THROW("Unsupported input {0}", *tensor.getDefiningOp());
}

// Check here that there are layers before ShapeCast
// which might be fused with Expand
// For now following cases are supported:
// 1.  PermuteQuantize -> Expand -> ShapeCase
//  in this case PermuteQuantize is fused with Expand later on
// 2. Slice -> DepthToSpace -> Expand -> ShapeCast
//  here Expand might be fused with DepthToSpace which has padded descriptor
// if we fuse these layers with Expand, ShapeCast uses already expanded tensor
// and reshape it back after Expand

bool EltwiseShapeCastRewriter::checkInput(mlir::Value tensor, int64_t origChannels) const {
    if (auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(tensor.getDefiningOp())) {
        // fusing with PermuteQuantize is availiable for VPUX37XX only
        const auto arch = VPU::getArch(permuteQuantize->getParentOfType<mlir::ModuleOp>());

        if (arch != VPU::ArchKind::VPUX37XX) {
            return false;
        }

        // PermuteQuantize is going to be fused with Expand only in that case
        if (getShape(permuteQuantize.output())[Dims4D::Act::C] != UNEXPANDED_CHANNELS &&
            origChannels != EXPANDED_CHANNELS) {
            return false;
        }

        return true;
    } else if (auto depthToSpace = mlir::dyn_cast_or_null<IE::DepthToSpaceOp>(tensor.getDefiningOp())) {
        if (!depthToSpace.output().hasOneUse()) {
            return false;
        }

        auto slice = depthToSpace.input().getDefiningOp<IE::SliceOp>();

        if (slice == nullptr || !slice.result().hasOneUse()) {
            return false;
        }

        if (getShape(slice.source())[Dims4D::Act::C] / (depthToSpace.block_size() * depthToSpace.block_size()) !=
            origChannels) {
            return false;
        }

        return true;
    }

    return false;
}

mlir::LogicalResult EltwiseShapeCastRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    // F16 case would be beneficial to be executed as Concat + Const.
    auto inputTensor = expandOp.input();
    if (inputTensor.getType()
                .cast<vpux::NDTypeInterface>()
                .getElementType()
                .dyn_cast_or_null<mlir::quant::QuantizedType>() == nullptr) {
        return mlir::failure();
    }

    auto lastShapeCastOp = inputTensor.getDefiningOp<IE::ShapeCastOp>();

    if (lastShapeCastOp == nullptr) {
        return mlir::failure();
    }

    auto addOp = lastShapeCastOp.source().getDefiningOp<IE::AddOp>();

    if (addOp == nullptr) {
        return mlir::failure();
    }

    auto firstInput = addOp.input1().getDefiningOp<IE::ShapeCastOp>();
    auto secondInput = addOp.input2().getDefiningOp<IE::ShapeCastOp>();

    if (firstInput == nullptr || secondInput == nullptr) {
        return mlir::failure();
    }

    auto origChannels = getShape(expandOp)[Dims4D::Act::C];
    bool equalInputs = firstInput == secondInput;

    if (!checkInput(firstInput.source(), origChannels) || !checkInput(secondInput.source(), origChannels)) {
        return mlir::failure();
    }

    Shape newUnexpandedShape = getShape(firstInput.source()).toValues();
    newUnexpandedShape[Dims4D::Act::C] = origChannels;
    Shape newExpandedShape = getShape(firstInput.source()).toValues();
    newExpandedShape[Dims4D::Act::C] = getShape(addOp)[Dims4D::Act::C];

    auto expandedShape = vpux::IE::getShapeCastExpandedShape(addOp.getOperation(), newExpandedShape, newUnexpandedShape,
                                                             _log.nest());

    if (mlir::failed(expandedShape)) {
        return mlir::failure();
    }

    auto expandedInput1 = getExpandedInput(rewriter, firstInput, expandOp, expandedShape.getValue());
    auto expandedInput2 =
            !equalInputs ? getExpandedInput(rewriter, secondInput, expandOp, expandedShape.getValue()) : expandedInput1;
    rewriter.setInsertionPoint(expandOp);

    auto outputType = expandedInput1.getType().cast<vpux::NDTypeInterface>().changeElemType(
            addOp.output().getType().cast<vpux::NDTypeInterface>().getElementType());
    auto newEltwise = rewriter.create<IE::AddOp>(addOp.getLoc(), outputType, expandedInput1, expandedInput2,
                                                 addOp.auto_broadcast(), nullptr);

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(expandOp, newEltwise.output(),
                                                 getIntArrayAttr(expandOp.getContext(), getShape(expandOp).raw()));

    return mlir::success();
}

//
// DepthToSpaceSliceRewriter
//

class DepthToSpaceSliceRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    DepthToSpaceSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("DepthToSpaceSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ExpandOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DepthToSpaceSliceRewriter::matchAndRewrite(IE::ExpandOp expandOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", expandOp->getName(), expandOp->getLoc());

    auto depthToSpace = expandOp.input().getDefiningOp<IE::DepthToSpaceOp>();

    if (depthToSpace == nullptr) {
        return mlir::failure();
    }

    auto slice = depthToSpace.input().getDefiningOp<IE::SliceOp>();

    if (slice == nullptr) {
        return mlir::failure();
    }

    auto sliceOffsets = parseIntArrayAttr<int64_t>(slice.static_offsets());
    auto hasNonZeroOffsets = llvm::any_of(sliceOffsets, [](auto offset) {
        return offset != 0;
    });
    if (hasNonZeroOffsets) {
        return mlir::failure();
    }

    if (!mlir::isa_and_nonnull<IE::AlignedChannelsOpInterface>(slice.source().getDefiningOp())) {
        return mlir::failure();
    }

    auto blockSizeSquare = depthToSpace.block_size() * depthToSpace.block_size();
    if (getShape(slice.source())[Dims4D::Act::C] / blockSizeSquare != getShape(expandOp)[Dims4D::Act::C]) {
        return mlir::failure();
    }

    auto paddedIC = getShape(slice.source())[Dims4D::Act::C] - getShape(slice.result())[Dims4D::Act::C];
    auto paddedOC = paddedIC / blockSizeSquare;

    auto paddedChannels = IE::ChannelPadding::get(getIntAttr(expandOp.getContext(), paddedIC),
                                                  getIntAttr(expandOp.getContext(), paddedOC), expandOp.getContext());

    rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(expandOp, slice.source(), depthToSpace.block_size(),
                                                    depthToSpace.mode(), paddedChannels);

    return mlir::success();
}

//
// SpaceToDepthSliceRewriter
//

class SpaceToDepthSliceRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    SpaceToDepthSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("SpaceToDepthSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createDPUOperation(mlir::PatternRewriter& rewriter, mlir::Value input, IE::SpaceToDepthOp s2dOp) const;
    void createPaddedConvolution(mlir::PatternRewriter& rewriter, mlir::Value input, IE::ConvolutionOp origConv) const;

    mlir::Value createConvFilter(mlir::PatternRewriter& rewriter, mlir::Value activation, int64_t blockSize) const;
    bool checkSliceExpand(IE::SliceOp sliceOp, IE::ExpandOp expandOp) const;
    Logger _log;
};

mlir::Value SpaceToDepthSliceRewriter::createConvFilter(mlir::PatternRewriter& rewriter, mlir::Value activation,
                                                        int64_t blockSize) const {
    const auto IC = getShape(activation)[Dims4D::Act::C];
    const auto KX = blockSize;
    const auto KY = blockSize;
    const auto OC = IC * blockSize * blockSize;

    const auto origElemType = activation.getType().cast<vpux::NDTypeInterface>().getElementType();
    const Shape weightShape = {OC, IC, KX, KY};

    mlir::Type filterElemType = mlir::Float16Type::get(getContext());
    if (auto qInputElemType = origElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto scale = 1.0f;
        const auto zeroPoint = 0;
        filterElemType = mlir::quant::UniformQuantizedType::get(
                0, getUInt8Type(getContext()), mlir::Float16Type::get(getContext()), scale, zeroPoint,
                std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    }

    auto filterTensorAttr = IE::getTensorAttr(getContext(), DimsOrder::OYXI, nullptr);
    auto filterType = mlir::RankedTensorType::get(weightShape.raw(), filterElemType, filterTensorAttr)
                              .cast<vpux::NDTypeInterface>();

    SmallVector<float> weights(weightShape.totalSize(), .0f);

    const auto outChannelStride = (IC * KX * KY);
    const auto inChannelStride = (KX * KY);
    const auto heightStride = KX;

    // set 1 to each filter so that it calculates the same as S2D
    // multiply coordinates on strides accordingly for each filter
    for (auto i = 0; i < OC; ++i) {
        const auto index = i * outChannelStride + (i % IC) * inChannelStride + (i / (KY * IC)) * heightStride +
                           ((i % (KY * IC)) / IC);
        weights[index] = 1.0f;
    }
    auto dataStorageType =
            mlir::RankedTensorType::get(weightShape.raw(), mlir::Float32Type::get(filterType.getContext()));
    auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(weights));

    auto contentAttr = Const::ContentAttr::get(dataAttr);
    if (auto qElemType = filterElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        contentAttr = contentAttr.convertElemType(getUInt8Type(filterType.getContext()));
        contentAttr = contentAttr.quantCast(qElemType);
    } else if (origElemType.isa<mlir::Float16Type>()) {
        contentAttr = contentAttr.convertElemType(mlir::Float16Type::get(filterType.getContext()));
    } else {
        VPUX_THROW("Unsupported type {0}", origElemType);
    }
    contentAttr = contentAttr.reorder(filterType.getDimsOrder());

    return rewriter.create<Const::DeclareOp>(activation.getLoc(), filterType, contentAttr);
}

mlir::Value SpaceToDepthSliceRewriter::createDPUOperation(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                          IE::SpaceToDepthOp s2dOp) const {
    const auto blockSize = s2dOp.block_size();
    const auto inputShape = getShape(input);
    const auto s2dOutShape = getShape(s2dOp.output());

    const auto convFilter = createConvFilter(rewriter, input, blockSize);

    auto newStrides = getIntArrayAttr(
            getContext(), ngraph::Strides{checked_cast<size_t>(blockSize), checked_cast<size_t>(blockSize)});
    auto newPadsBegin = getIntArrayAttr(getContext(), ngraph::CoordinateDiff{0, 0});
    auto newPadsEnd = getIntArrayAttr(getContext(), ngraph::CoordinateDiff{0, 0});
    auto newDilations = getIntArrayAttr(getContext(), ngraph::Strides{1, 1});

    const Shape outPadBefore(checked_cast<size_t>(s2dOp.getType().getRank()), 0);

    Shape outPadAfter(checked_cast<size_t>(s2dOp.getType().getRank()), 0);
    outPadAfter[Dims4D::Act::C] = inputShape[Dims4D::Act::C] * blockSize * blockSize - s2dOutShape[Dims4D::Act::C];

    const auto ndType = s2dOp.getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

    return rewriter.create<IE::ConvolutionOp>(s2dOp.getLoc(), newOutputType, input, convFilter, nullptr, newStrides,
                                              newPadsBegin, newPadsEnd, newDilations, nullptr);
}

void SpaceToDepthSliceRewriter::createPaddedConvolution(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                        IE::ConvolutionOp origConv) const {
    auto filter = origConv.filter();

    auto filterConst = filter.getDefiningOp<Const::DeclareOp>();

    const auto filterShape = getShape(filter);
    const auto channelsPad = getShape(input)[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];

    const auto newFilterShape = Shape({filterShape[Dims4D::Filter::OC], getShape(input)[Dims4D::Act::C],
                                       filterShape[Dims4D::Filter::KX], filterShape[Dims4D::Filter::KY]});
    auto newContentAttr =
            filterConst.contentAttr()
                    .reshape({filterShape[Dims4D::Filter::OC], filterShape[Dims4D::Filter::IC] / channelsPad,
                              filterShape[Dims4D::Filter::KX] * channelsPad, filterShape[Dims4D::Filter::KY]})
                    .padWithZero({0, 0, 0, 0}, {0, 1, 0, 0})
                    .reshape(newFilterShape);

    const auto origFilterType = filter.getType().cast<vpux::NDTypeInterface>();
    const auto outAllocType =
            mlir::RankedTensorType::get(to_small_vector(newFilterShape), origFilterType.getElementType())
                    .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto paddedFilter = rewriter.create<Const::DeclareOp>(filterConst.getLoc(), outAllocTypeNHWC, newContentAttr);

    mlir::BlockAndValueMapping mapper;
    mapper.map(origConv.getOperands(), makeArrayRef({input, paddedFilter.output()}));
    auto* newOperand = rewriter.clone(*origConv, mapper);

    rewriter.replaceOp(origConv, newOperand->getResult(0));
}

// Check if we have case when
// PermuteQuantizeOp -> ExpandOp (4 channels) -> SliceOp (3 channels) -> SpaceToDepth
bool SpaceToDepthSliceRewriter::checkSliceExpand(IE::SliceOp sliceOp, IE::ExpandOp expandOp) const {
    if (expandOp.input().getDefiningOp<IE::PermuteQuantizeOp>() == nullptr) {
        return false;
    }

    const auto expandShape = getShape(expandOp.output());
    const auto sliceShape = getShape(sliceOp.result());

    if (expandShape[Dims4D::Act::C] != EXPANDED_CHANNELS || sliceShape[Dims4D::Act::C] != UNEXPANDED_CHANNELS) {
        return false;
    }

    if (expandOp.input().getType() != sliceOp.result().getType()) {
        return false;
    }

    auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.static_offsets());
    auto hasNonZeroOffsets = llvm::any_of(sliceOffsets, [](auto offset) {
        return offset != 0;
    });
    if (hasNonZeroOffsets) {
        return false;
    }

    return true;
}

// Convert pattern
//   PermuteQuantize                                 PermuteQuantize
//        |                                                 |
//     Expand (4 channels)                              Expand (4 channels)
//        |                                                 |
//      Slice (3 channels)           ->              Convolution (S2D on DPU)
//        |                                                 |
//    SpaceToDepth                                   Convolution (padded filter)
//        |
//    Convolution
mlir::LogicalResult SpaceToDepthSliceRewriter::matchAndRewrite(IE::ConvolutionOp convOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", convOp->getName(), convOp->getLoc());

    auto spaceToDepth = convOp.input().getDefiningOp<IE::SpaceToDepthOp>();

    if (spaceToDepth == nullptr || !spaceToDepth.output().hasOneUse()) {
        return mlir::failure();
    }

    // for now only block_first case supported
    // E#69685 for depth_first mode
    if (spaceToDepth.mode() != IE::SpaceToDepthMode::BLOCKS_FIRST) {
        return mlir::failure();
    }

    auto slice = spaceToDepth.input().getDefiningOp<IE::SliceOp>();

    if (slice == nullptr) {
        return mlir::failure();
    }

    auto expandOp = slice.source().getDefiningOp<IE::ExpandOp>();

    if (expandOp == nullptr) {
        return mlir::failure();
    }

    if (!checkSliceExpand(slice, expandOp)) {
        return mlir::failure();
    }

    auto s2dDPU = createDPUOperation(rewriter, expandOp.output(), spaceToDepth);

    createPaddedConvolution(rewriter, s2dDPU, convOp);

    return mlir::success();
}

//
// PropagateExpandPass
//

class PropagateExpandPass final : public IE::PropagateExpandBase<PropagateExpandPass> {
public:
    explicit PropagateExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void PropagateExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EltwiseShapeCastRewriter>(&ctx, _log);
    patterns.add<DepthToSpaceSliceRewriter>(&ctx, _log);
    patterns.add<SpaceToDepthSliceRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateExpandPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateExpandPass(Logger log) {
    return std::make_unique<PropagateExpandPass>(log);
}

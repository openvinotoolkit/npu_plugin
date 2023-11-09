//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/deconvolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// GroupDeconvConverter
//

class GroupDeconvConverter final : public mlir::OpRewritePattern<IE::GroupDeconvolutionOp> {
public:
    GroupDeconvConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupDeconvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupDeconvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupDeconvConverter::matchAndRewrite(IE::GroupDeconvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got GroupDeconvolutionOp layer at '{0}'", origOp->getLoc());

    auto padsOutput = Shape(parseIntArrayAttr<int64_t>(origOp.output_padding()));

    const auto featureShape = getShape(origOp.feature());
    if (featureShape.size() != 4) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupDeconvolutionOp is supported, expected 4D feature but got {0}",
                           featureShape.size());
    }

    const auto outputShape = getShape(origOp.output());
    if (outputShape.size() != 4) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupDeconvolutionOp is supported, expected 4D output shape but got {0}",
                           outputShape.size());
    }

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto origFilterShape = to_small_vector(filterType.getShape());
    if (origFilterShape.size() != 5) {
        return matchFailed(rewriter, origOp,
                           "Only 2D GroupDeconvolutionOp is supported, expected 5D filter but got {0}",
                           origFilterShape.size());
    }

    auto dwConvFilter = origOp.filter().getDefiningOp<Const::DeclareOp>();
    if (dwConvFilter == nullptr) {
        return matchFailed(rewriter, origOp, "[Unsupported]Filter is not Const::DeclareOp at {0}", origOp->getLoc());
    }

    auto featureUpScale = IE::createUpsampling(rewriter, origOp, padsOutput, true);
    if (mlir::failed(featureUpScale)) {
        _log.nest().trace("Failed to create Upsampling for {0}", origOp->getLoc());
        return mlir::failure();
    }
    auto paddingOutput = featureUpScale.value();

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    const auto elemType = filterType.getElementType();
    const auto filterContentAttr = dwConvFilter.getContentAttr();

    std::swap(origFilterShape[IE::GROUP_DECONV_C_OUT_DIM_INDEX], origFilterShape[IE::GROUP_DECONV_C_IN_DIM_INDEX]);
    const auto dataStorageType = mlir::RankedTensorType::get(origFilterShape, elemType);

    SmallVector<unsigned int> permVec = {0, IE::GROUP_DECONV_C_OUT_DIM_INDEX, IE::GROUP_DECONV_C_IN_DIM_INDEX,
                                         IE::GROUP_DECONV_KY_DIM_INDEX, IE::GROUP_DECONV_KX_DIM_INDEX};
    auto perm = mlir::AffineMap::getPermutationMap(permVec, getContext());
    const auto content = filterContentAttr.reverse(Dim(IE::GROUP_DECONV_C_OUT_DIM_INDEX))
                                 .convertElemType(elemType)
                                 .transpose(DimsOrder::fromAffineMap(perm));
    auto reshapedFilter = rewriter.create<Const::DeclareOp>(dwConvFilter.getLoc(), dataStorageType, content);

    // convert filter shape from 5D to 4D
    auto groups = origFilterShape[0];
    origFilterShape[1] *= groups;
    origFilterShape.erase(origFilterShape.begin());

    const auto filter4DShapeAttr = getIntArrayAttr(rewriter.getContext(), origFilterShape);
    auto reshaped4DFilter = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), reshapedFilter.getOutput(), nullptr,
                                                                 false, filter4DShapeAttr);

    const auto postOp = origOp.post_opAttr();

    if (padsOutput[Dims4D::PadsOutput::Y] > 0) {
        paddingOutput = IE::createPadding(rewriter, origOp->getLoc(), paddingOutput, Dims4D::Act::H,
                                          padsOutput[Dims4D::PadsOutput::Y], nullptr);
    }
    if (padsOutput[Dims4D::PadsOutput::X] > 0) {
        paddingOutput = IE::createPadding(rewriter, origOp->getLoc(), paddingOutput, Dims4D::Act::W,
                                          padsOutput[Dims4D::PadsOutput::X], nullptr);
    }

    auto resultOP =
            rewriter.create<IE::GroupConvolutionOp>(origOp->getLoc(), paddingOutput, reshaped4DFilter, nullptr, strides,
                                                    padsBegin, padsEnd, dilations, getIntAttr(rewriter, groups), postOp)
                    .output();

    rewriter.replaceOp(origOp, resultOP);

    _log.trace("Replaced GroupDeconvolutionOp at '{0}' with 'IE::GroupConvolutionOp' (2D)", origOp.getLoc());

    return mlir::success();
}

//
// ConvertGroupDeconvToGroupConvPass
//

class ConvertGroupDeconvToGroupConvPass final :
        public IE::ConvertGroupDeconvToGroupConvBase<ConvertGroupDeconvToGroupConvPass> {
public:
    explicit ConvertGroupDeconvToGroupConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertGroupDeconvToGroupConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalGroupDeconv = [&](IE::GroupDeconvolutionOp groupDeconv) {
        _log.trace("Got '{0}' at '{1}'", groupDeconv->getName(), groupDeconv->getLoc());
        if (mlir::failed(IE::canConvertGroupDeconvToGroupConv(groupDeconv))) {
            _log.nest(1).trace("GroupDeconvolutionOp cannot be converted.");
            return true;
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::GroupDeconvolutionOp>(isLegalGroupDeconv);
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupDeconvConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGroupDeconvToGroupConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupDeconvToGroupConvPass(Logger log) {
    return std::make_unique<ConvertGroupDeconvToGroupConvPass>(log);
}

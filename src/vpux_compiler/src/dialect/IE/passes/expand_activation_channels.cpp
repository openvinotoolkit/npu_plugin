//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes/expand_activation_channels.hpp"

using namespace vpux;

namespace vpux {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape) {
    Shape padsEnd(origShape.size());

    for (auto i : irange(origShape.size())) {
        const auto d = Dim(i);
        padsEnd[d] = extendedShape[d] - origShape[d];
    }

    return padsEnd;
}

Shape calcPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    extendedShape[Dims4D::Act::C] = alignValUp(origShape[Dims4D::Act::C], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

//
// generalRewrite
//

// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input
//

mlir::LogicalResult generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator, Logger log) {
    auto* ctx = origOp->getContext();

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp);

    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    const auto inPadsEnd = calcPadsEnd(inputType, iface.getInputChannelAlignment());
    const auto outPadsEnd = calcPadsEnd(outputType, iface.getOutputChannelAlignment());

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[Dims4D::Act::C] == 0 && outPadsEnd[Dims4D::Act::C] == 0) {
        return matchFailed(log, rewriter, origOp, "Both input and output channels are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput =
                rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(0), None, ShapeRef(inPadsEnd));
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, outPadsEnd[Dims4D::Act::C]);

    if (outPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        const SmallVector<int64_t> offsets(outShape.size(), 0);

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp->getResult(0).getType(), newOp->getResult(0),
                                                 getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));
    }

    return mlir::success();
}

}  // namespace vpux

//
// MaxPoolRewriter
//

mlir::LogicalResult IE::MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.kernel_size(),
                                              origOp.strides(), origOp.pads_begin(), origOp.pads_end(),
                                              origOp.rounding_type(), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// ConvolutionRewriter
//

mlir::Value concatWithZeroConst(mlir::Location loc, mlir::Value filter, ShapeRef subInput,
                                mlir::PatternRewriter& rewriter) {
    const auto filterType = filter.getType().cast<vpux::NDTypeInterface>();

    auto padShape = to_small_vector(filterType.getShape());
    padShape[Dims4D::Filter::IC.ind()] = subInput[Dims4D::Filter::IC];

    auto const generateZeroConst = [&]() {
        const auto padType = filterType.changeShape(ShapeRef(padShape));
        const auto eleType = padType.getElementType();

        const auto getEleStorageType = [&]() {
            if (const auto quantizedType = eleType.dyn_cast<mlir::quant::QuantizedType>()) {
                return normalizeQuantStorageType(quantizedType);
            } else {
                return eleType;
            }
        };
        const auto storageElementType = getEleStorageType();

        auto outputBuffer = Const::Content::allocTempBuffer(padType, storageElementType, false);
        outputBuffer.fillWithZero();

        const auto dataType = padType.changeElemType(storageElementType).cast<mlir::RankedTensorType>();
        mlir::DenseElementsAttr eleAttr;
        const auto getDataAttr = [&](auto buffer) {
            eleAttr = mlir::DenseElementsAttr::get(dataType, buffer);
        };
        outputBuffer.mutate(getDataAttr);

        return rewriter.create<Const::DeclareOp>(loc, padType, Const::ContentAttr::get(eleAttr)).getOutput();
    };

    SmallVector<mlir::Value> concatInput;
    concatInput.push_back(filter);
    concatInput.push_back(generateZeroConst());
    auto concatOp = rewriter.create<IE::ConcatOp>(loc, concatInput, Dims4D::Filter::IC);

    return concatOp.output();
}

//
// ConvolutionRewriter
//

mlir::LogicalResult IE::ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto filterShape = getShape(origOp.filter());

        const auto newInputShape = getShape(expandedInput);
        const auto inChanPadEnd = newInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];

        mlir::Value paddedFilter;

        if (inChanPadEnd == 0 && outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;
            filterPadsEnd[Dims4D::Filter::IC] = inChanPadEnd;

            auto filterOp = origOp.filter().getDefiningOp();

            bool isConstFilter = mlir::isa_and_nonnull<Const::DeclareOp>(filterOp);
            if (!isConstFilter) {
                if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(filterOp)) {
                    const auto fqInputConstOp = fqOp.input().getDefiningOp<Const::DeclareOp>();
                    isConstFilter = fqInputConstOp != nullptr;
                }
            }

            const auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
            bool isFp16Type = filterType.getElementType().isa<mlir::Float16Type>();

            // E#72287: Convert ExpandOp to const Concat in VPUIP, ExpandOp is preferred in IE for optimization.
            const auto expandTensor = [&](mlir::Value filter, ShapeRef pad) {
                if (isFp16Type) {
                    return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), filter, None, pad);
                } else {
                    return concatWithZeroConst(origOp->getLoc(), filter, pad, rewriter);
                }
            };

            if (!isConstFilter && inChanPadEnd != 0 && outChanPadEnd == 0) {
                // 1 dim expand for non-const filter
                _log.trace("Pad non-const filter in IC at '{0}'", origOp->getLoc());
                paddedFilter = expandTensor(origOp.filter(), filterPadsEnd);

            } else if (!isConstFilter && inChanPadEnd != 0 && outChanPadEnd != 0) {
                // 2 dims expand for non-const filter
                _log.trace("Pad non-const filter in IC & OC at '{0}'", origOp->getLoc());

                mlir::Value paddedFilter1;
                Shape filterPadsEnd1(filterShape.size(), 0);
                filterPadsEnd1[Dims4D::Filter::IC] = inChanPadEnd;
                paddedFilter1 = expandTensor(origOp.filter(), filterPadsEnd1);

                Shape filterPadsEnd2(filterShape.size(), 0);
                filterPadsEnd2[Dims4D::Filter::OC] = outChanPadEnd;
                paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), paddedFilter1, None,
                                                                   ShapeRef(filterPadsEnd2));
            } else {
                // Const filter expand or expand on OC only
                paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.filter(), None,
                                                                   ShapeRef(filterPadsEnd));
            }
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.bias(), None,
                                                                   ShapeRef(biasPadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), newOutputType, expandedInput, paddedFilter,
                                                  paddedBiases, origOp.strides(), origOp.pads_begin(),
                                                  origOp.pads_end(), origOp.dilations(), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// GroupConvolutionRewriter
//

mlir::LogicalResult IE::GroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolutionOp layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        const auto filterShape = getShape(origOp.filter());

        mlir::Value paddedFilter;

        if (outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;

            paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.filter(), None,
                                                               ShapeRef(filterPadsEnd));
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.bias(), None,
                                                                   ShapeRef(biasPadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);
        const auto newConvOutShape = newOutputType.getShape().toValues();

        return rewriter.create<IE::GroupConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, paddedBiases, origOp.strides(),
                origOp.pads_begin(), origOp.pads_end(), origOp.dilations(),
                getIntAttr(getContext(), newConvOutShape[Dims4D::Act::C]), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// InterpolateRewriter
//

mlir::LogicalResult IE::InterpolateRewriter::matchAndRewrite(IE::InterpolateOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Interpolate layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::InterpolateOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.sizes(),
                                                  origOp.scales(), origOp.axes(), origOp.sizes_attrAttr(),
                                                  origOp.scales_attrAttr(), origOp.axes_attrAttr(),
                                                  origOp.tile_offset_attrAttr(), origOp.initial_input_dims_attrAttr(),
                                                  origOp.initial_output_dims_attrAttr(), origOp.attrAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

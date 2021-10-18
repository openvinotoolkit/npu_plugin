//
// Copyright Intel Corporation.
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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// allocateResults
//

SmallVector<mlir::Value> allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::TypeConverter& typeConverter, mlir::ValueRange origResults) {
    return to_small_vector(origResults | transformed([&](mlir::Value origVal) -> mlir::Value {
                               auto origType = origVal.getType();
                               auto memRefType = typeConverter.convertType(origType);
                               auto allocOp =
                                       builder.create<mlir::memref::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
                               return allocOp.memref();
                           }));
}

//
// ReshapeRewrite
//

class ReshapeRewrite final : public mlir::OpInterfaceConversionPattern<mlir::ViewLikeOpInterface> {
public:
    ReshapeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceConversionPattern<mlir::ViewLikeOpInterface>(typeConverter, ctx), _log(log) {
        setDebugName("ReshapeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReshapeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    if (!mlir::isa<IE::ReshapeOp, IE::SqueezeOp, IE::UnsqueezeOp>(origOp)) {
        return mlir::failure();
    }

    _log.trace("Found Reshape Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    if (!outType.hasStaticShape()) {
        return matchFailed(rewriter, origOp, "GenericReshape with dynamic shape is not supported yet");
    }

    VPUX_THROW_UNLESS(!newOperands.empty(), "Got wrong newOperands size : '{0}'", newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    rewriter.replaceOpWithNewOp<IERT::GenericReshapeOp>(origOp, newOutType, newOperands[0]);
    return mlir::success();
}

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpConversionPattern<IE::SplitOp> {
public:
    SplitRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SplitOp>(typeConverter, ctx), _log(log) {
        setDebugName("SplitRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitRewrite::matchAndRewrite(IE::SplitOp origOp, ArrayRef<mlir::Value> newOperands,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(),
                      "Got wrong newOperands size : '{0}', expected '{1}'", newOperands.size(),
                      origOp->getNumOperands());

    if (!origOp.axis_value().hasValue()) {
        return matchFailed(rewriter, origOp, "Got non constant axis");
    }

    const auto inputType = newOperands[0].getType().cast<mlir::ShapedType>();
    const auto inputShape = getShape(inputType);

    const auto axis = Dim(origOp.axis_value().getValue());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.getResults());

    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> results;

    const auto offsetStep = inputShape[axis] / origOp.num_splits();

    for (auto i : irange(origOp->getNumResults())) {
        const auto origOutputType = origOp.getResult(i).getType().cast<mlir::ShapedType>();
        const auto svSizes = origOutputType.getShape();

        _log.trace("Create SubView for output #'{0}'", i);
        auto subView = rewriter.create<IERT::SubViewOp>(origOp.getLoc(), newOperands[0], svOffsets, svSizes);

        _log.trace("Copy SubView result to output buffer");

        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), subView, allocatedBufs[i]);
        results.push_back(copyOp.output());

        svOffsets[axis.ind()] += offsetStep;
    }

    rewriter.replaceOp(origOp, results);

    return mlir::success();
}

//
// ConcatRewrite
//

class ConcatRewrite final : public mlir::OpConversionPattern<IE::ConcatOp> {
public:
    ConcatRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConcatOp>(typeConverter, ctx), _log(log) {
        setDebugName("ConcatRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatRewrite::matchAndRewrite(IE::ConcatOp origOp, ArrayRef<mlir::Value> newOperands,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Concat Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(),
                      "Got wrong newOperands size : '{0}', expected '{1}'", newOperands.size(),
                      origOp->getNumOperands());

    const auto axis = Dim(origOp.axis());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, {origOp.getResult()});

    const auto outputRank = origOp.getType().getRank();

    SmallVector<int64_t> svOffsets(outputRank, 0);
    SmallVector<mlir::Value> results;

    for (auto i : irange(origOp->getNumOperands())) {
        const auto newInputType = newOperands[i].getType().cast<mlir::ShapedType>();
        const auto svSizes = newInputType.getShape();

        _log.trace("Create SubView for input #'{0}'", i);
        auto subView = rewriter.create<IERT::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes);

        _log.trace("Copy new operand to SubView");
        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), newOperands[i], subView);
        results.push_back(copyOp.output());

        svOffsets[axis.ind()] += svSizes[axis.ind()];
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, results, allocatedBufs[0]);
    return mlir::success();
}

//
// SubTensorRewrite
//

class SubTensorRewrite final : public mlir::OpConversionPattern<IE::SliceOp> {
public:
    SubTensorRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::SliceOp>(typeConverter, ctx), _log(log) {
        setDebugName("SubTensorRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SubTensorRewrite::matchAndRewrite(IE::SliceOp origOp, ArrayRef<mlir::Value> newOperands,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found SubTensor Operation '{0}'", origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(),
                      "Got wrong newOperands size : '{0}', expected '{1}'", newOperands.size(),
                      origOp->getNumOperands());

    auto subView = rewriter.create<IERT::SubViewOp>(origOp->getLoc(), newOperands[0], origOp.static_offsetsAttr(),
                                                    origOp.static_sizesAttr());

    auto allocatedBuf = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.getResult());

    auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), subView, allocatedBuf[0]);

    rewriter.replaceOp(origOp, {copyOp});
    return mlir::success();
}

//
// ExpandRewrite
//

class ExpandRewrite final : public mlir::OpConversionPattern<IE::ExpandOp> {
public:
    ExpandRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ExpandOp>(typeConverter, ctx), _log(log) {
        setDebugName("ExpandRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandRewrite::matchAndRewrite(IE::ExpandOp origOp, ArrayRef<mlir::Value> newOperands,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Expand Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "ExpandRewrite: failed to get type converter");

    auto expandedBuffer = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.output());
    const auto inputType = newOperands[0].getType().cast<mlir::ShapedType>();

    auto subOffsetsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    auto subShape = to_small_vector(inputType.getShape());

    const auto chunk = subShape[IE::Dims4D::Act::C.ind()];
    const auto OC = getShape(origOp.output())[IE::Dims4D::Act::C];

    SmallVector<mlir::Value> concatInputs;
    const auto fullCopyNum = OC / chunk;

    // The first version copied the input once. Example:
    // tensor<1x3xHxWxf16>(first channel:[0.1, 0.2, 0.3]) -> Expand ->
    // tensor<1x8xHxWxf16>(first channel:[0.1, 0.2, 0.3, val1, val2, val3, val4, val5])
    // It was assumed that zero weights would allow not to take into account "garbage" values
    // (val1, val2, ...) falling into the tensor during expansion.

    // It turned out that after some calculations, Inf/NaN can remain in memory.
    // IEEE 754: Infinities propagate through calculations; NaN infects any calculation that involves it.
    // Now assuming that the input contains only valid values.
    // Fill in these values the space that appeared after the channels were expanded. Example:
    // tensor<1x3xHxWxf16>(first channel:[0.1, 0.2, 0.3]) -> Expand ->
    // tensor<1x8xHxWxf16>(first channel:[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2])

    for (int copyIdx = 0; copyIdx < fullCopyNum; copyIdx++) {
        auto subView = rewriter.create<IERT::SubViewOp>(origOp.getLoc(), expandedBuffer[0], subOffsetsBegin, subShape);
        auto subViewCopy = rewriter.create<IERT::CopyOp>(origOp->getLoc(), newOperands[0], subView);

        concatInputs.push_back(subViewCopy.output());

        subOffsetsBegin[IE::Dims4D::Act::C.ind()] += chunk;
    }

    const auto filledSize = subOffsetsBegin[IE::Dims4D::Act::C.ind()];
    if (filledSize < OC) {
        SmallVector<int64_t> subInputOffsetsBegin{0, 0, 0, 0};
        subShape[IE::Dims4D::Act::C.ind()] = OC - filledSize;

        auto subViewInput =
                rewriter.create<IERT::SubViewOp>(origOp.getLoc(), newOperands[0], subInputOffsetsBegin, subShape);
        auto subViewTail =
                rewriter.create<IERT::SubViewOp>(origOp.getLoc(), expandedBuffer[0], subOffsetsBegin, subShape);

        auto subViewCopy = rewriter.create<IERT::CopyOp>(origOp->getLoc(), subViewInput, subViewTail);

        concatInputs.push_back(subViewCopy.output());
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, concatInputs, expandedBuffer[0]);

    return mlir::success();
}

//
// LSTMCellRewrite
//

class LSTMCellRewrite final : public mlir::OpConversionPattern<IE::LSTMCellOp> {
public:
    LSTMCellRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::LSTMCellOp>(typeConverter, ctx), _log(log) {
        setDebugName("LSTMCellRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(IE::LSTMCellOp origOp, ArrayRef<mlir::Value> newOperands,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMCell Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    IE::LSTMCellOp::Adaptor args(newOperands, origOp->getAttrDictionary());

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), args.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.recurrenceWeights().getType(), args.recurrenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<IE::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 1);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<IERT::LSTMCellOp>(origOp, args.inputData(), args.initialHiddenState(),
                                                  args.initialCellState(), targetConcatenatedWeights, args.biases(),
                                                  resultBufs[0], resultBufs[1], origOp.hiddenSizeAttr());

    return mlir::success();
}

//
// LayerRewrite
//

mlir::Operation* createRTLayer(IE::QuantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::QuantizeOp::Adaptor newOp(allBufs);
    return b.create<IERT::QuantizeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::DequantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::DequantizeOp::Adaptor newOp(allBufs);
    return b.create<IERT::DequantizeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::ConvertOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ConvertOp::Adaptor newOp(allBufs);
    return b.create<IERT::ConvertOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::CopyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CopyOp::Adaptor newOp(allBufs);
    return b.create<IERT::CopyOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

// static int tt = 0;

mlir::Operation* createRTLayer(IE::ReadValueOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    std::cout << "ReadValue operation convertion to IERT" << std::endl;

    IERT::ReadValueOp::Adaptor newOp(allBufs);

    auto input = newOp.input();
    // auto second_input = newOp.second_input();
    // auto output = newOp.output();
    std::cout << "ReadValue constructor" << std::endl;

    // auto p = b.create<IERT::ReadValueOp>(origOp.getLoc(), input, input, newOp.output_buff(), origOp.variable_id());
    auto p = b.create<IERT::ReadValueOp>(origOp.getLoc(), input, input, newOp.output_buff(), origOp.variable_id());

    // auto p = b.create<IERT::ReadValueOp>(origOp.getLoc(), newOp.input(), newOp.second_input(), newOp.output_buff(), origOp.variable_id());
    // auto p = b.create<IERT::ReadValueOp>(origOp.getLoc(), newOp.input(), nullptr, newOp.output_buff(), origOp.variable_id());
    std::cout << "ReadValue operation convertion to IERT END" << std::endl;
    return p;
}

mlir::Operation* createRTLayer(IE::AssignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AssignOp::Adaptor newOp(allBufs);

    std::cout << "Assign operation convertion to IERT" << std::endl;

    return b.create<IERT::AssignOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.variable_id());
}

mlir::Operation* createRTLayer(IE::ReLUOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReLUOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReLUOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SigmoidOp::Adaptor newOp(allBufs);
    return b.create<IERT::SigmoidOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::HSwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::HSwishOp::Adaptor newOp(allBufs);
    return b.create<IERT::HSwishOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::FloorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FloorOp::Adaptor newOp(allBufs);
    return b.create<IERT::FloorOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::MishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MishOp::Adaptor newOp(allBufs);
    return b.create<IERT::MishOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::ErfOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ErfOp::Adaptor newOp(allBufs);
    return b.create<IERT::ErfOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::TanhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TanhOp::Adaptor newOp(allBufs);
    return b.create<IERT::TanhOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::NegativeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::NegativeOp::Adaptor newOp(allBufs);
    return b.create<IERT::NegativeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::PReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PReluOp::Adaptor newOp(allBufs);
    return b.create<IERT::PReluOp>(origOp.getLoc(), newOp.input(), newOp.negative_slope(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::AddOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AddOp::Adaptor newOp(allBufs);
    return b.create<IERT::AddOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                 origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::MultiplyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MultiplyOp::Adaptor newOp(allBufs);
    return b.create<IERT::MultiplyOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::DivideOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::DivideOp::Adaptor newOp(allBufs);
    return b.create<IERT::DivideOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SquaredDifferenceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SquaredDifferenceOp::Adaptor newOp(allBufs);
    return b.create<IERT::SquaredDifferenceOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::PowerOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PowerOp::Adaptor newOp(allBufs);
    return b.create<IERT::PowerOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::FloorModOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FloorModOp::Adaptor newOp(allBufs);
    return b.create<IERT::FloorModOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::MinimumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MinimumOp::Adaptor newOp(allBufs);
    return b.create<IERT::MinimumOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::MaximumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MaximumOp::Adaptor newOp(allBufs);
    return b.create<IERT::MaximumOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::TileOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TileOp::Adaptor newOp(allBufs);
    return b.create<IERT::TileOp>(origOp.getLoc(), newOp.input(), newOp.repeats(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SoftMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SoftMaxOp::Adaptor newOp(allBufs);
    return b.create<IERT::SoftMaxOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axisIndAttr());
}

mlir::Operation* createRTLayer(IE::AvgPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AvgPoolOp::Adaptor newOp(allBufs);
    return b.create<IERT::AvgPoolOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.kernel_sizeAttr(),
                                     origOp.stridesAttr(), origOp.pads_beginAttr(), origOp.pads_endAttr(),
                                     origOp.exclude_padsAttr());
}

mlir::Operation* createRTLayer(IE::MaxPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MaxPoolOp::Adaptor newOp(allBufs);
    return b.create<IERT::MaxPoolOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.kernel_sizeAttr(),
                                     origOp.stridesAttr(), origOp.pads_beginAttr(), origOp.pads_endAttr(),
                                     origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::ClampOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ClampOp::Adaptor newOp(allBufs);
    return b.create<IERT::ClampOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.minAttr(),
                                   origOp.maxAttr());
}

mlir::Operation* createRTLayer(IE::EluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::EluOp::Adaptor newOp(allBufs);
    return b.create<IERT::EluOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.xAttr());
}

mlir::Operation* createRTLayer(IE::LeakyReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LeakyReluOp::Adaptor newOp(allBufs);
    return b.create<IERT::LeakyReluOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                       origOp.negative_slopeAttr());
}

mlir::Operation* createRTLayer(IE::GRNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GRNOp::Adaptor newOp(allBufs);
    return b.create<IERT::GRNOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.biasAttr());
}

mlir::Operation* createRTLayer(IE::PerAxisTileOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PerAxisTileOp::Adaptor newOp(allBufs);
    return b.create<IERT::PerAxisTileOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axisAttr(),
                                         origOp.tilesAttr());
}

mlir::Operation* createRTLayer(IE::FakeQuantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FakeQuantizeOp::Adaptor newOp(allBufs);
    return b.create<IERT::FakeQuantizeOp>(origOp.getLoc(), newOp.input(), newOp.input_low(), newOp.input_high(),
                                          newOp.output_low(), newOp.output_high(), newOp.output_buff(),
                                          origOp.levelsAttr());
}

mlir::Operation* createRTLayer(IE::ROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ROIPoolingOp::Adaptor newOp(allBufs);
    return b.create<IERT::ROIPoolingOp>(origOp.getLoc(), newOp.input(), newOp.coords(), newOp.output_buff(),
                                        origOp.output_sizeAttr(), origOp.spatial_scaleAttr(), origOp.methodAttr());
}

mlir::Operation* createRTLayer(IE::ConvolutionOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ConvolutionOp::Adaptor newOp(allBufs);
    return b.create<IERT::ConvolutionOp>(origOp.getLoc(), newOp.input(), newOp.filter(), newOp.bias(),
                                         newOp.output_buff(), origOp.stridesAttr(), origOp.pads_beginAttr(),
                                         origOp.pads_endAttr(), origOp.dilationsAttr(), origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::GroupConvolutionOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GroupConvolutionOp::Adaptor newOp(allBufs);
    return b.create<IERT::GroupConvolutionOp>(origOp.getLoc(), newOp.input(), newOp.filter(), newOp.bias(),
                                              newOp.output_buff(), origOp.stridesAttr(), origOp.pads_beginAttr(),
                                              origOp.pads_endAttr(), origOp.dilationsAttr(), origOp.groupsAttr(),
                                              origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::SwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SwishOp::Adaptor newOp(allBufs);
    return b.create<IERT::SwishOp>(origOp.getLoc(), newOp.input(), newOp.beta(), newOp.output_buff(),
                                   origOp.beta_valueAttr());
}

mlir::Operation* createRTLayer(IE::FullyConnectedOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FullyConnectedOp::Adaptor newOp(allBufs);
    return b.create<IERT::FullyConnectedOp>(origOp.getLoc(), newOp.input(), newOp.weights(), newOp.bias(),
                                            newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::DetectionOutputOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newPreds = origOp.in_additional_preds() != nullptr ? allBufs[3] : nullptr;
    const auto newProposals = origOp.in_additional_proposals() != nullptr ? allBufs[4] : nullptr;
    return b.create<IERT::DetectionOutputOp>(origOp->getLoc(), allBufs[0], allBufs[1], allBufs[2], newPreds,
                                             newProposals, allBufs.back(), origOp.attr());
}

mlir::Operation* createRTLayer(IE::ScaleShiftOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    mlir::Value newWeights;
    mlir::Value newBiases;
    if (origOp.weights() != nullptr && origOp.biases() != nullptr) {
        newWeights = allBufs[1];
        newBiases = allBufs[2];
    } else if (origOp.weights() != nullptr) {
        newWeights = allBufs[1];
    } else if (origOp.biases() != nullptr) {
        newBiases = allBufs[1];
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }
    return b.create<IERT::ScaleShiftOp>(origOp->getLoc(), allBufs[0], newWeights, newBiases, allBufs.back());
}

mlir::Operation* createRTLayer(IE::TransposeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TransposeOp::Adaptor newOp(allBufs);
    return b.create<IERT::TransposeOp>(origOp.getLoc(), newOp.input(), newOp.order(), newOp.output_buff(),
                                       origOp.order_valueAttr());
}

mlir::Operation* createRTLayer(IE::CTCGreedyDecoderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CTCGreedyDecoderOp::Adaptor newOp(allBufs);
    return b.create<IERT::CTCGreedyDecoderOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLengths(),
                                              newOp.output_buff(), origOp.mergeRepeatedAttr());
}

mlir::Operation* createRTLayer(IE::CTCGreedyDecoderSeqLenOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CTCGreedyDecoderSeqLenOp::Adaptor newOp(allBufs);
    return b.create<IERT::CTCGreedyDecoderSeqLenOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLength(),
                                                    newOp.blankIndex(), newOp.output_buff(), newOp.outputLength_buff(),
                                                    origOp.mergeRepeatedAttr());
}

mlir::Operation* createRTLayer(IE::PadOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(origOp.pads_begin_attr().hasValue() && origOp.pads_end_attr().hasValue(),
                      "PadOp must use attributes for `pads_begin` and `pads_end` params");

    IERT::PadOp::Adaptor newOp(allBufs);
    return b.create<IERT::PadOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                 origOp.pads_begin_attr().getValue(), origOp.pads_end_attr().getValue(),
                                 origOp.pad_value_attrAttr(), origOp.mode());
}

mlir::Operation* createRTLayer(IE::ExpOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ExpOp::Adaptor newOp(allBufs);
    return b.create<IERT::ExpOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::InterpolateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(origOp.sizes_attr().hasValue() && origOp.scales_attr().hasValue(),
                      "Interpolate must have constant sizes or scales");
    VPUX_THROW_UNLESS(origOp.axes_attr().hasValue(), "Interpolate must have constant axes");

    IERT::InterpolateOp::Adaptor newOp(allBufs);
    return b.create<IERT::InterpolateOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                         origOp.attr().mode().getValue(), origOp.attr().coord_mode().getValue(),
                                         origOp.attr().nearest_mode().getValue(), origOp.attr().antialias().getValue());
}

mlir::Operation* createRTLayer(IE::StridedSliceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(allBufs.size() == 2, "Constant inputs should have been converted to attributes");
    VPUX_THROW_UNLESS(origOp.begins_attr().hasValue(), "begins_attr is null");
    VPUX_THROW_UNLESS(origOp.ends_attr().hasValue(), "ends_attr is null");
    VPUX_THROW_UNLESS(origOp.strides_attr().hasValue(), "strides_attr is null");

    return b.create<IERT::StridedSliceOp>(origOp.getLoc(), allBufs[0], allBufs.back(), origOp.begins_attr().getValue(),
                                          origOp.ends_attr().getValue(), origOp.strides_attr().getValue());
}

mlir::Operation* createRTLayer(IE::ReorderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReorderOp::Adaptor newOp(allBufs);

    return b.create<IERT::ReorderOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::RegionYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::RegionYoloOp::Adaptor newOp(allBufs);
    return b.create<IERT::RegionYoloOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.coords(),
                                        origOp.classes(), origOp.regions(), origOp.do_softmax(), origOp.mask(),
                                        origOp.axis(), origOp.end_axis(), origOp.anchors());
}

mlir::Operation* createRTLayer(IE::MVNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MVNOp::Adaptor newOp(allBufs);
    return b.create<IERT::MVNOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.across_channels(),
                                 origOp.normalize_variance(), origOp.eps());
}
mlir::Operation* createRTLayer(IE::SubtractOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SubtractOp::Adaptor newOp(allBufs);
    return b.create<IERT::SubtractOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

class LayerRewrite final : public mlir::ConversionPattern {
public:
    LayerRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}, benefitLow, ctx), _log(log) {
        setDebugName("LayerRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    template <class InLayerOp>
    static mlir::Operation* dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b);

private:
    Logger _log;
};

template <class InLayerOp>
mlir::Operation* LayerRewrite::dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    return createRTLayer(mlir::cast<InLayerOp>(origOp), allBufs, b);
}

mlir::LogicalResult LayerRewrite::matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    using CreateFunc =
            mlir::Operation* (*)(mlir::Operation * origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder & b);

#define CASE(_OP_)                                   \
    .Case<_OP_>([](mlir::Operation*) -> CreateFunc { \
        return dispatch<_OP_>;                       \
    })

    const CreateFunc createFunc = llvm::TypeSwitch<mlir::Operation*, CreateFunc>(origOp)  //
            CASE(IE::QuantizeOp)
    CASE(IE::DequantizeOp)
    CASE(IE::ConvertOp)
    CASE(IE::SoftMaxOp)
    CASE(IE::AvgPoolOp)
    CASE(IE::MaxPoolOp)
    CASE(IE::ConvolutionOp)
    CASE(IE::GroupConvolutionOp)
    CASE(IE::ReLUOp)
    CASE(IE::SigmoidOp)
    CASE(IE::ClampOp)
    CASE(IE::EluOp)
    CASE(IE::HSwishOp)
    CASE(IE::FloorOp)
    CASE(IE::MishOp)
    CASE(IE::ErfOp)
    CASE(IE::TanhOp)
    CASE(IE::FakeQuantizeOp)
    CASE(IE::PReluOp)
    CASE(IE::LeakyReluOp)
    CASE(IE::AddOp)
    CASE(IE::MultiplyOp)
    CASE(IE::DivideOp)
    CASE(IE::SquaredDifferenceOp)
    CASE(IE::PowerOp)
    CASE(IE::FloorModOp)
    CASE(IE::MinimumOp)
    CASE(IE::MaximumOp)
    CASE(IE::SwishOp)
    CASE(IE::GRNOp)
    CASE(IE::TileOp)
    CASE(IE::PerAxisTileOp)
    CASE(IE::NegativeOp)
    CASE(IE::ROIPoolingOp)
    CASE(IE::FullyConnectedOp)
    CASE(IE::DetectionOutputOp)
    CASE(IE::ScaleShiftOp)
    CASE(IE::TransposeOp)
    CASE(IE::CTCGreedyDecoderOp)
    CASE(IE::CTCGreedyDecoderSeqLenOp)
    CASE(IE::PadOp)
    CASE(IE::ExpOp)
    CASE(IE::InterpolateOp)
    CASE(IE::StridedSliceOp)
    CASE(IE::ReorderOp)
    CASE(IE::RegionYoloOp)
    CASE(IE::MVNOp)
    CASE(IE::SubtractOp)
    CASE(IE::ReadValueOp)
    CASE(IE::AssignOp)
    CASE(IE::CopyOp)
    .Default([](mlir::Operation*) {
        return nullptr;
    });

#undef CASE

    if (createFunc == nullptr) {
        return mlir::failure();
    }

    _log.trace("Found Layer Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(), "Got wrong newOperands size : '{0}'",
                      newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto resultBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    SmallVector<mlir::Value> allBufs;
    allBufs.reserve(newOperands.size() + resultBufs.size());
    allBufs.append(newOperands.begin(), newOperands.end());
    allBufs.append(resultBufs.begin(), resultBufs.end());

    const auto newOp = createFunc(origOp, allBufs, rewriter);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// BufferizeIEPass
//

class BufferizeIEPass final : public BufferizeIEBase<BufferizeIEPass> {
public:
    explicit BufferizeIEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BufferizeIEPass::safeRunOnFunc() {
    std::cout << "BufferizeIEPass::safeRunOnFunc()" << std::endl;

    auto& ctx = getContext();

    vpux::BufferizeTypeConverter typeConverter;

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<Const::ConstDialect>(isLegalOp);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ReshapeRewrite>(typeConverter, &ctx, _log);
    patterns.insert<SplitRewrite>(typeConverter, &ctx, _log);
    patterns.insert<ConcatRewrite>(typeConverter, &ctx, _log);
    patterns.insert<LayerRewrite>(typeConverter, &ctx, _log);
    patterns.insert<SubTensorRewrite>(typeConverter, &ctx, _log);
    patterns.insert<ExpandRewrite>(typeConverter, &ctx, _log);
    patterns.insert<LSTMCellRewrite>(typeConverter, &ctx, _log);
    Const::ConstDialect::populateBufferizePatterns(patterns, typeConverter, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createBufferizeIEPass
//

std::unique_ptr<mlir::Pass> vpux::createBufferizeIEPass(Logger log) {
    return std::make_unique<BufferizeIEPass>(log);
}

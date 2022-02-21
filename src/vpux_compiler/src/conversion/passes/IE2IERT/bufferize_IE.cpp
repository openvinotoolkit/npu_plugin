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
#include "vpux/compiler/core/layers.hpp"
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

template <class ConcreteOp>
class ReshapeRewrite final : public mlir::OpConversionPattern<ConcreteOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<ConcreteOp>::OpAdaptor;

public:
    ReshapeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ConcreteOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("ReshapeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ReshapeRewrite<ConcreteOp>::matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Reshape Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp.getType();

    if (!outType.hasStaticShape()) {
        return matchFailed(rewriter, origOp, "'{0}' with dynamic shape is not supported yet",
                           IERT::GenericReshapeOp::getOperationName());
    }

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    rewriter.replaceOpWithNewOp<IERT::GenericReshapeOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

//
// PermuteCastRewrite
//

class PermuteCastRewrite final : public mlir::OpConversionPattern<IE::PermuteCastOp> {
public:
    PermuteCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::PermuteCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("PermuteCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PermuteCastRewrite::matchAndRewrite(IE::PermuteCastOp origOp, OpAdaptor newArgs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found PermuteCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<IERT::PermuteCastOp>(origOp, newOutType, newArgs.input(), origOp.dst_orderAttr(),
                                                     origOp.mem_permAttr());
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
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitRewrite::matchAndRewrite(IE::SplitOp origOp, OpAdaptor newArgs,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    if (!origOp.axis_value().hasValue()) {
        return matchFailed(rewriter, origOp, "Got non constant axis");
    }

    const auto inputType = newArgs.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    const auto axis = Dim(origOp.axis_value().getValue());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.getResults());

    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> results;

    const auto offsetStep = inputShape[axis] / origOp.num_splits();

    for (auto i : irange(origOp->getNumResults())) {
        const auto origOutputType = origOp.getResult(i).getType().cast<vpux::NDTypeInterface>();
        const auto svSizes = origOutputType.getShape().raw();

        _log.trace("Create SubView for output #'{0}'", i);
        auto subView = rewriter.create<IERT::SubViewOp>(origOp.getLoc(), newArgs.input(), svOffsets, svSizes);

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
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> rewriteWithAxis(IE::ConcatOp origOp, OpAdaptor newArgs,
                                             ArrayRef<mlir::Value> allocatedBufs,
                                             mlir::ConversionPatternRewriter& rewriter) const;
    SmallVector<mlir::Value> rewriteWithOffsets(IE::ConcatOp origOp, OpAdaptor newArgs,
                                                ArrayRef<mlir::Value> allocatedBufs,
                                                mlir::ConversionPatternRewriter& rewriter) const;

private:
    Logger _log;
};

SmallVector<mlir::Value> ConcatRewrite::rewriteWithAxis(IE::ConcatOp origOp, OpAdaptor newArgs,
                                                        ArrayRef<mlir::Value> allocatedBufs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    SmallVector<mlir::Value> results;

    const auto axis = origOp.per_axisAttr().axis().getValue().getSExtValue();
    const auto offset = origOp.per_axisAttr().offset() ? origOp.per_axisAttr().offset().getValue().getSExtValue() : 0;
    const auto stride = origOp.per_axisAttr().stride() ? origOp.per_axisAttr().stride().getValue().getSExtValue() : 1;

    const auto outputRank = origOp.getType().getRank();

    SmallVector<int64_t> svOffsets(outputRank, 0);

    SmallVector<int64_t> svElemStrides;
    if (stride != 1) {
        svElemStrides.resize(outputRank, 1);
        svElemStrides[axis] = stride;
    }

    for (auto i : irange(origOp->getNumOperands())) {
        const auto newInput = newArgs.inputs()[i];
        const auto newInputType = newInput.getType().cast<vpux::NDTypeInterface>();
        const auto svSizes = newInputType.getShape().raw();

        _log.trace("Create SubView for input #'{0}'", i);
        mlir::Value subViewVal;
        if (svElemStrides.empty()) {
            subViewVal = rewriter.create<IERT::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes);
            svOffsets[axis] += svSizes[axis];
        } else {
            subViewVal = rewriter.create<IERT::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes,
                                                          svElemStrides);
            svOffsets[axis] += offset;
        }

        _log.trace("Copy new operand to SubView");
        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), newInput, subViewVal);
        results.push_back(copyOp.output());
    }

    return results;
}

SmallVector<mlir::Value> ConcatRewrite::rewriteWithOffsets(IE::ConcatOp origOp, OpAdaptor newArgs,
                                                           ArrayRef<mlir::Value> allocatedBufs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    SmallVector<mlir::Value> results;

    const auto allOffsets = origOp.static_offsetsAttr().getAsRange<mlir::ArrayAttr>();

    for (const auto p : zip(newArgs.inputs(), allOffsets)) {
        const auto newInput = std::get<0>(p);

        const auto curShape = newInput.getType().cast<vpux::NDTypeInterface>().getShape().raw();
        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));

        auto subViewOp = rewriter.create<IERT::SubViewOp>(origOp->getLoc(), allocatedBufs[0], curOffsets, curShape);

        auto copyOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), newInput, subViewOp.result());
        results.push_back(copyOp.output());
    }

    return results;
}

mlir::LogicalResult ConcatRewrite::matchAndRewrite(IE::ConcatOp origOp, OpAdaptor newArgs,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Concat Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, {origOp.getResult()});

    const auto results = origOp.per_axisAttr() ? rewriteWithAxis(origOp, newArgs, allocatedBufs, rewriter)
                                               : rewriteWithOffsets(origOp, newArgs, allocatedBufs, rewriter);

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
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SubTensorRewrite::matchAndRewrite(IE::SliceOp origOp, OpAdaptor newArgs,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found SubTensor Operation '{0}'", origOp->getLoc());

    auto subView = rewriter.create<IERT::SubViewOp>(origOp->getLoc(), newArgs.source(), origOp.static_offsetsAttr(),
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
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandRewrite::matchAndRewrite(IE::ExpandOp origOp, OpAdaptor newArgs,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Expand Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "ExpandRewrite: failed to get type converter");

    auto expandedBuffer = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp.output());
    const auto inputType = newArgs.input().getType().cast<vpux::NDTypeInterface>();

    auto subOffsetsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    auto subShape = to_small_vector(inputType.getShape());

    const auto chunk = subShape[Dims4D::Act::C.ind()];
    const auto OC = getShape(origOp.output())[Dims4D::Act::C];

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
        auto subViewCopy = rewriter.create<IERT::CopyOp>(origOp->getLoc(), newArgs.input(), subView);

        concatInputs.push_back(subViewCopy.output());

        subOffsetsBegin[Dims4D::Act::C.ind()] += chunk;
    }

    const auto filledSize = subOffsetsBegin[Dims4D::Act::C.ind()];
    if (filledSize < OC) {
        SmallVector<int64_t> subInputOffsetsBegin{0, 0, 0, 0};
        subShape[Dims4D::Act::C.ind()] = OC - filledSize;

        auto subViewInput =
                rewriter.create<IERT::SubViewOp>(origOp.getLoc(), newArgs.input(), subInputOffsetsBegin, subShape);
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
    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(IE::LSTMCellOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMCell Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), newArgs.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.recurrenceWeights().getType(), newArgs.recurrenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<IE::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 1);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<IERT::LSTMCellOp>(
            origOp, newArgs.inputData(), newArgs.initialHiddenState(), newArgs.initialCellState(),
            targetConcatenatedWeights, newArgs.biases(), resultBufs[0], resultBufs[1], origOp.hiddenSizeAttr());

    return mlir::success();
}

//
// LSTMSequenceRewrite
//

class LSTMSequenceRewrite final : public mlir::OpConversionPattern<IE::LSTMSequenceOp> {
public:
    LSTMSequenceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::LSTMSequenceOp>(typeConverter, ctx), _log(log) {
        setDebugName("LSTMSequenceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(IE::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), newArgs.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.reccurenceWeights().getType(), newArgs.reccurenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<IE::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 2);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<IERT::LSTMSequenceOp>(origOp, newArgs.inputData(), newArgs.initialHiddenState(),
                                                      newArgs.initialCellState(), targetConcatenatedWeights,
                                                      newArgs.biases(), resultBufs[0], resultBufs[1], resultBufs[2],
                                                      origOp.sequenceLengthAttr(), origOp.directionAttr());

    return mlir::success();
}

//
// ReverseSequenceRewrite
//

class ReverseSequenceRewrite final : public mlir::OpConversionPattern<IE::ReverseSequenceOp> {
public:
    ReverseSequenceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ReverseSequenceOp>(typeConverter, ctx), _log(log) {
        setDebugName("ReverseSequenceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReverseSequenceRewrite::matchAndRewrite(IE::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found ReverseSequence Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto origSeqLengthShapeType = origOp.seq_length().getType().cast<mlir::ShapedType>();
    auto newSeqLengthShapeType =
            origSeqLengthShapeType.clone(origSeqLengthShapeType.getShape(), mlir::Float16Type::get(getContext()));
    auto memRefType = typeConverter->convertType(newSeqLengthShapeType);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), memRefType.cast<mlir::MemRefType>());

    auto convertOp = rewriter.create<IERT::ConvertOp>(origOp->getLoc(), newArgs.seq_length(), allocOp.memref());

    auto resultBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<IERT::ReverseSequenceOp>(origOp, newArgs.data(), convertOp.output(), resultBufs[0],
                                                         origOp.seq_axisAttr(), origOp.batch_axisAttr());

    return mlir::success();
}

//
// QuantizeCastRewriter
//

class QuantizeCastRewriter final : public mlir::OpConversionPattern<IE::QuantizeCastOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<IE::QuantizeCastOp>::OpAdaptor;

public:
    QuantizeCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::QuantizeCastOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("QuantizeCastRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeCastRewriter::matchAndRewrite(IE::QuantizeCastOp origOp, OpAdaptor newArgs,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found QuantizeCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp.getType();

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    rewriter.replaceOpWithNewOp<IERT::QuantizeCastOp>(origOp, newOutType, newArgs.input());
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

mlir::Operation* createRTLayer(IE::ReLUOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReLUOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReLUOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SigmoidOp::Adaptor newOp(allBufs);
    return b.create<IERT::SigmoidOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SignOp::Adaptor newOp(allBufs);
    return b.create<IERT::SignOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::HSwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::HSwishOp::Adaptor newOp(allBufs);
    return b.create<IERT::HSwishOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::FloorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::FloorOp::Adaptor newOp(allBufs);
    return b.create<IERT::FloorOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::RoundOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::RoundOp::Adaptor newOp(allBufs);
    return b.create<IERT::RoundOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.mode());
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

mlir::Operation* createRTLayer(IE::SqrtOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SqrtOp::Adaptor newOp(allBufs);
    return b.create<IERT::SqrtOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SinhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SinhOp::Adaptor newOp(allBufs);
    return b.create<IERT::SinhOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::CoshOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CoshOp::Adaptor newOp(allBufs);
    return b.create<IERT::CoshOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::AsinhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AsinhOp::Adaptor newOp(allBufs);
    return b.create<IERT::AsinhOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::AcoshOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AcoshOp::Adaptor newOp(allBufs);
    return b.create<IERT::AcoshOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::LogOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LogOp::Adaptor newOp(allBufs);
    return b.create<IERT::LogOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::GeluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GeluOp::Adaptor newOp(allBufs);
    return b.create<IERT::GeluOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::NegativeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::NegativeOp::Adaptor newOp(allBufs);
    return b.create<IERT::NegativeOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::PReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::PReluOp::Adaptor newOp(allBufs);
    return b.create<IERT::PReluOp>(origOp.getLoc(), newOp.input(), newOp.negative_slope(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::GatherOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GatherOp::Adaptor newOp(allBufs);
    return b.create<IERT::GatherOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.axis(), newOp.output_buff(),
                                    origOp.axis_valueAttr(), origOp.batch_dimsAttr());
}

mlir::Operation* createRTLayer(IE::YuvToRgbOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newInp2 = origOp.input2() != nullptr ? allBufs[2 - 1] : nullptr;
    const auto newInp3 = origOp.input3() != nullptr ? allBufs[3 - 1] : nullptr;
    return b.create<IERT::YuvToRgbOp>(origOp.getLoc(), allBufs[0], newInp2, newInp3, allBufs.back(), origOp.inFmtAttr(),
                                      origOp.outFmtAttr());
}

mlir::Operation* createRTLayer(IE::GatherElementsOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GatherElementsOp::Adaptor newOp(allBufs);
    return b.create<IERT::GatherElementsOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.output_buff(),
                                            origOp.axisAttr());
}

mlir::Operation* createRTLayer(IE::ScatterNDUpdateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ScatterNDUpdateOp::Adaptor newOp(allBufs);
    return b.create<IERT::ScatterNDUpdateOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.updates(),
                                             newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::AddOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AddOp::Adaptor newOp(allBufs);
    return b.create<IERT::AddOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                 origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::MultiplyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MultiplyOp::Adaptor newOp(allBufs);
    return b.create<IERT::MultiplyOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                      origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::AndOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::AndOp::Adaptor newOp(allBufs);
    return b.create<IERT::AndOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                 origOp.post_opAttr());
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

mlir::Operation* createRTLayer(IE::LRN_IEOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LRN_IEOp::Adaptor newOp(allBufs);
    return b.create<IERT::LRN_IEOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.alphaAttr(),
                                    origOp.betaAttr(), origOp.biasAttr(), origOp.sizeAttr(), origOp.regionAttr());
}

mlir::Operation* createRTLayer(IE::BroadcastOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::BroadcastOp::Adaptor newOp(allBufs);
    return b.create<IERT::BroadcastOp>(origOp.getLoc(), newOp.input(), newOp.target_shape(), newOp.axes_mapping(),
                                       newOp.output_buff(), origOp.modeAttr());
}

mlir::Operation* createRTLayer(IE::ReduceMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReduceMaxOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReduceMaxOp>(origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(),
                                       origOp.keep_dimsAttr());
}

mlir::Operation* createRTLayer(IE::ReduceMeanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReduceMeanOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReduceMeanOp>(origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(),
                                        origOp.keep_dimsAttr());
}

mlir::Operation* createRTLayer(IE::ReduceSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReduceSumOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReduceSumOp>(origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(),
                                       origOp.keep_dimsAttr());
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

mlir::Operation* createRTLayer(IE::ROIAlignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ROIAlignOp::Adaptor newOp(allBufs);
    return b.create<IERT::ROIAlignOp>(
            origOp.getLoc(), newOp.input(), newOp.coords(), newOp.roisIdx(), newOp.output_buff(), origOp.pooled_hAttr(),
            origOp.pooled_wAttr(), origOp.sampling_ratioAttr(), origOp.spatial_scaleAttr(), origOp.poolingModeAttr());
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

mlir::Operation* createRTLayer(IE::RegionYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::RegionYoloOp::Adaptor newOp(allBufs);
    return b.create<IERT::RegionYoloOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.coords(),
                                        origOp.classes(), origOp.regions(), origOp.do_softmax(), origOp.mask(),
                                        origOp.axis(), origOp.end_axis(), origOp.anchors());
}

mlir::Operation* createRTLayer(IE::ReorgYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::ReorgYoloOp::Adaptor newOp(allBufs);
    return b.create<IERT::ReorgYoloOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.strideAttr());
}

mlir::Operation* createRTLayer(IE::MVNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MVNOp::Adaptor newOp(allBufs);
    return b.create<IERT::MVNOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.across_channels(),
                                 origOp.normalize_variance(), origOp.eps());
}

mlir::Operation* createRTLayer(IE::DepthToSpaceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::DepthToSpaceOp::Adaptor newOp(allBufs);
    return b.create<IERT::DepthToSpaceOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.block_sizeAttr(),
                                          origOp.modeAttr());
}

mlir::Operation* createRTLayer(IE::SubtractOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SubtractOp::Adaptor newOp(allBufs);
    return b.create<IERT::SubtractOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                      origOp.post_opAttr());
}

mlir::Operation* createRTLayer(IE::MemPermuteOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::MemPermuteOp::Adaptor newOp(allBufs);

    return b.create<IERT::MemPermuteOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.mem_perm());
}

mlir::Operation* createRTLayer(IE::SoftPlusOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SoftPlusOp::Adaptor newOp(allBufs);
    return b.create<IERT::SoftPlusOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::CeilingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CeilingOp::Adaptor newOp(allBufs);
    return b.create<IERT::CeilingOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::NormalizeIEOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::NormalizeIEOp::Adaptor newOp(allBufs);
    return b.create<IERT::NormalizeIEOp>(origOp.getLoc(), newOp.data(), newOp.weights(), newOp.output_buff(),
                                         origOp.eps(), origOp.across_spatial(), origOp.channel_shared());
}

mlir::Operation* createRTLayer(IE::EqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::EqualOp::Adaptor newOp(allBufs);
    return b.create<IERT::EqualOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::UpsamplingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::UpsamplingOp::Adaptor newOp(allBufs);
    return b.create<IERT::UpsamplingOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                        origOp.upsampling_factorAttr(), origOp.pad_lAttr(), origOp.pad_rAttr());
}

mlir::Operation* createRTLayer(IE::LessOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LessOp::Adaptor newOp(allBufs);
    return b.create<IERT::LessOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::LessEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LessEqualOp::Adaptor newOp(allBufs);
    return b.create<IERT::LessEqualOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::TopKOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::TopKOp::Adaptor newOp(allBufs);
    return b.create<IERT::TopKOp>(origOp.getLoc(), newOp.input(), newOp.k(), newOp.output_values_buff(),
                                  newOp.target_shape_buff(), origOp.axisAttr(), origOp.modeAttr(), origOp.sortAttr(),
                                  origOp.element_typeAttr());
}

mlir::Operation* createRTLayer(IE::NotEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::NotEqualOp::Adaptor newOp(allBufs);
    return b.create<IERT::NotEqualOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::GreaterOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GreaterOp::Adaptor newOp(allBufs);
    return b.create<IERT::GreaterOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::GreaterEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::GreaterEqualOp::Adaptor newOp(allBufs);
    return b.create<IERT::GreaterEqualOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::LogicalNotOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LogicalNotOp::Adaptor newOp(allBufs);
    return b.create<IERT::LogicalNotOp>(origOp.getLoc(), newOp.input1(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::LogicalOrOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LogicalOrOp::Adaptor newOp(allBufs);
    return b.create<IERT::LogicalOrOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::LogicalXorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::LogicalXorOp::Adaptor newOp(allBufs);
    return b.create<IERT::LogicalXorOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff());
}

mlir::Operation* createRTLayer(IE::SpaceToDepthOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::SpaceToDepthOp::Adaptor newOp(allBufs);
    return b.create<IERT::SpaceToDepthOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.block_size(),
                                          origOp.mode());
}

mlir::Operation* createRTLayer(IE::CopyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CopyOp::Adaptor newOp(allBufs);
    return b.create<IERT::CopyOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
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
    CASE(IE::SignOp)
    CASE(IE::ClampOp)
    CASE(IE::EluOp)
    CASE(IE::HSwishOp)
    CASE(IE::FloorOp)
    CASE(IE::RoundOp)
    CASE(IE::MishOp)
    CASE(IE::ErfOp)
    CASE(IE::BroadcastOp)
    CASE(IE::ReduceMaxOp)
    CASE(IE::ReduceMeanOp)
    CASE(IE::ReduceSumOp)
    CASE(IE::TanhOp)
    CASE(IE::SqrtOp)
    CASE(IE::SinhOp)
    CASE(IE::CoshOp)
    CASE(IE::AsinhOp)
    CASE(IE::AcoshOp)
    CASE(IE::LogOp)
    CASE(IE::GeluOp)
    CASE(IE::FakeQuantizeOp)
    CASE(IE::PReluOp)
    CASE(IE::GatherOp)
    CASE(IE::YuvToRgbOp)
    CASE(IE::GatherElementsOp)
    CASE(IE::ScatterNDUpdateOp)
    CASE(IE::LeakyReluOp)
    CASE(IE::AddOp)
    CASE(IE::MultiplyOp)
    CASE(IE::AndOp)
    CASE(IE::DivideOp)
    CASE(IE::SquaredDifferenceOp)
    CASE(IE::PowerOp)
    CASE(IE::FloorModOp)
    CASE(IE::MinimumOp)
    CASE(IE::MaximumOp)
    CASE(IE::SwishOp)
    CASE(IE::GRNOp)
    CASE(IE::LRN_IEOp)
    CASE(IE::TileOp)
    CASE(IE::PerAxisTileOp)
    CASE(IE::NegativeOp)
    CASE(IE::ROIPoolingOp)
    CASE(IE::ROIAlignOp)
    CASE(IE::FullyConnectedOp)
    CASE(IE::DetectionOutputOp)
    CASE(IE::ScaleShiftOp)
    CASE(IE::CTCGreedyDecoderOp)
    CASE(IE::CTCGreedyDecoderSeqLenOp)
    CASE(IE::PadOp)
    CASE(IE::ExpOp)
    CASE(IE::InterpolateOp)
    CASE(IE::StridedSliceOp)
    CASE(IE::RegionYoloOp)
    CASE(IE::ReorgYoloOp)
    CASE(IE::MVNOp)
    CASE(IE::DepthToSpaceOp)
    CASE(IE::SubtractOp)
    CASE(IE::MemPermuteOp)
    CASE(IE::SoftPlusOp)
    CASE(IE::CeilingOp)
    CASE(IE::NormalizeIEOp)
    CASE(IE::EqualOp)
    CASE(IE::UpsamplingOp)
    CASE(IE::LessOp)
    CASE(IE::LessEqualOp)
    CASE(IE::NotEqualOp)
    CASE(IE::GreaterOp)
    CASE(IE::GreaterEqualOp)
    CASE(IE::SpaceToDepthOp)
    CASE(IE::TopKOp)
    CASE(IE::LogicalNotOp)
    CASE(IE::LogicalOrOp)
    CASE(IE::LogicalXorOp)
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
    patterns.add<ReshapeRewrite<IE::AffineReshapeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<IE::ReshapeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<IE::SqueezeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<IE::UnsqueezeOp>>(typeConverter, &ctx, _log);
    patterns.add<SplitRewrite>(typeConverter, &ctx, _log);
    patterns.add<ConcatRewrite>(typeConverter, &ctx, _log);
    patterns.add<LayerRewrite>(typeConverter, &ctx, _log);
    patterns.add<SubTensorRewrite>(typeConverter, &ctx, _log);
    patterns.add<ExpandRewrite>(typeConverter, &ctx, _log);
    patterns.add<LSTMCellRewrite>(typeConverter, &ctx, _log);
    patterns.add<LSTMSequenceRewrite>(typeConverter, &ctx, _log);
    patterns.add<PermuteCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<QuantizeCastRewriter>(typeConverter, &ctx, _log);
    patterns.add<ReverseSequenceRewrite>(typeConverter, &ctx, _log);
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

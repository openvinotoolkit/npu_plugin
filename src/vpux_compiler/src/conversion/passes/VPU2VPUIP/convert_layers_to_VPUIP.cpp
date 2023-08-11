//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {
mlir::OpResult createCopyResult(mlir::Type type, mlir::Value inputBuffer, mlir::Value outputBuffer,
                                mlir::ConversionPatternRewriter& rewriter, mlir::Location location) {
    if (type == nullptr) {
        return mlir::OpResult();
    }

    auto dataType = type;
    if (auto sparseBuffer = dataType.dyn_cast<VPUIP::SparseBufferType>()) {
        dataType = sparseBuffer.getData();
    }

    if (dataType.isa<mlir::MemRefType, VPUIP::BufferType>()) {
        auto copyOp = rewriter.create<VPUIP::CopyOp>(location, inputBuffer, outputBuffer);

        return copyOp.getOperation()->getResult(0);
    } else if (dataType.isa<VPUIP::DistributedBufferType>()) {
        // Create NCEClusterTiling with CopyOp inside
        SmallVector<mlir::Value> inputsOutputOperands = {inputBuffer, outputBuffer};

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        auto clusterTilingOp =
                rewriter.create<VPUIP::NCEClusterTilingOp>(location, type, inputsOutputOperands, bodyBuilder);
        return clusterTilingOp.getOperation()->getResult(0);
    }
    VPUX_THROW("Unexpected data type to copy: {0}", dataType);
}

//
// CopyRewrite
//

class CopyRewrite final : public mlir::OpConversionPattern<VPU::CopyOp> {
public:
    CopyRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::CopyOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("StridedSliceRewrite");
    }

    mlir::LogicalResult matchAndRewrite(VPU::CopyOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyRewrite::matchAndRewrite(VPU::CopyOp origOp, OpAdaptor newArgs,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found VPU::Copy Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());
    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(origOp, newArgs.input(), outputBuffers[0]);

    return mlir::success();
}

//
// ExpandRewrite
//

class ExpandRewrite final : public mlir::OpConversionPattern<VPU::ExpandOp> {
public:
    ExpandRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ExpandOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("ExpandRewrite");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ExpandOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandRewrite::matchAndRewrite(VPU::ExpandOp origOp, OpAdaptor newArgs,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found VPU::Expand Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());
    rewriter.replaceOpWithNewOp<VPUIP::ExpandOp>(origOp, newArgs.input(), outputBuffers[0], origOp.pads_begin(),
                                                 origOp.pads_end());
    return mlir::success();
}

//
// StridedSliceRewrite
//

class StridedSliceRewrite final : public mlir::OpConversionPattern<VPU::StridedSliceOp> {
public:
    StridedSliceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::StridedSliceOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("StridedSliceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::StridedSliceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StridedSliceRewrite::matchAndRewrite(VPU::StridedSliceOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found VPU::StridedSlice Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto origType = origOp.getType();
    const auto newOutType = typeConverter->convertType(origType);

    const auto outShape = getShape(origOp.output());
    auto outShapeAttr = getIntArrayAttr(rewriter, outShape.raw());
    auto subView = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), newArgs.input(), origOp.begins_attrAttr(),
                                                     outShapeAttr, origOp.strides_attrAttr());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    auto newResult = createCopyResult(newOutType, subView.result(), outputBuffers[0], rewriter, origOp->getLoc());

    rewriter.replaceOp(origOp, newResult);

    return mlir::success();
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

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

//
// SliceRewrite
//

class SliceRewrite final : public mlir::OpConversionPattern<VPU::SliceOp> {
public:
    SliceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::SliceOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("SliceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SliceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SliceRewrite::matchAndRewrite(VPU::SliceOp origOp, OpAdaptor newArgs,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found VPU::Slice Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto origType = origOp.getType();
    const auto newOutType = typeConverter->convertType(origType);

    auto subView = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), newArgs.source(), origOp.static_offsetsAttr(),
                                                     origOp.static_sizesAttr());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    mlir::OpResult newResult =
            createCopyResult(newOutType, subView.result(), outputBuffers[0], rewriter, origOp->getLoc());

    rewriter.replaceOp(origOp, newResult);

    return mlir::success();
}

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpConversionPattern<VPU::SplitOp> {
public:
    SplitRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::SplitOp>(typeConverter, ctx), _log(log) {
        setDebugName("SplitRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SplitOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitRewrite::matchAndRewrite(VPU::SplitOp origOp, OpAdaptor newArgs,
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

    auto allocatedBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp.getResults());

    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> results;

    const auto offsetStep = inputShape[axis] / origOp.num_splits();

    for (auto i : irange(origOp->getNumResults())) {
        const auto origOutputType = origOp.getResult(i).getType().cast<vpux::NDTypeInterface>();
        const auto svSizes = origOutputType.getShape().raw();

        _log.trace("Create SubView for output #'{0}'", i);
        auto subView = rewriter.create<VPUIP::SubViewOp>(origOp.getLoc(), newArgs.input(), svOffsets, svSizes);

        _log.trace("Copy SubView result to output buffer");

        auto copyOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), subView, allocatedBufs[i]);
        results.push_back(copyOp.output());

        svOffsets[axis.ind()] += offsetStep;
    }

    rewriter.replaceOp(origOp, results);

    return mlir::success();
}

//
// ConcatRewrite
//

class ConcatRewrite final : public mlir::OpConversionPattern<VPU::ConcatOp> {
public:
    ConcatRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ConcatOp>(typeConverter, ctx), _log(log) {
        setDebugName("ConcatRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> rewriteWithAxis(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                             ArrayRef<mlir::Value> allocatedBufs,
                                             mlir::ConversionPatternRewriter& rewriter) const;
    SmallVector<mlir::Value> rewriteWithOffsets(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                ArrayRef<mlir::Value> allocatedBufs,
                                                mlir::ConversionPatternRewriter& rewriter) const;

private:
    Logger _log;
};

SmallVector<mlir::Value> ConcatRewrite::rewriteWithAxis(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                        ArrayRef<mlir::Value> allocatedBufs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    SmallVector<mlir::Value> results;

    const auto axis = origOp.per_axisAttr().axis().getValue().getSExtValue();
    const auto offset = origOp.per_axisAttr().offset() ? origOp.per_axisAttr().offset().getValue().getSExtValue() : 0;
    const auto stride = origOp.per_axisAttr().stride() ? origOp.per_axisAttr().stride().getValue().getSExtValue() : 1;

    const auto outputRank = origOp.getType().cast<vpux::NDTypeInterface>().getRank();

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
            subViewVal = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes);
            svOffsets[axis] += svSizes[axis];
        } else {
            subViewVal = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes,
                                                           svElemStrides);
            svOffsets[axis] += offset;
        }

        _log.trace("Copy new operand to SubView");

        auto newOutType = subViewVal.getType();

        // Copy to the SubView
        mlir::OpResult newResult = createCopyResult(newOutType, newInput, subViewVal, rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

SmallVector<mlir::Value> ConcatRewrite::rewriteWithOffsets(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                           ArrayRef<mlir::Value> allocatedBufs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    SmallVector<mlir::Value> results;

    const auto allOffsets = origOp.static_offsetsAttr().getAsRange<mlir::ArrayAttr>();

    for (const auto p : zip(newArgs.inputs(), allOffsets)) {
        const auto newInput = std::get<0>(p);

        const auto curShape = newInput.getType().cast<vpux::NDTypeInterface>().getShape().raw();
        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));

        _log.trace("Create SubView");

        auto subViewOp = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), allocatedBufs[0], curOffsets, curShape);

        _log.trace("Copy new operand to SubView");

        auto newOutType = subViewOp.result().getType();

        // Copy to the SubView
        mlir::OpResult newResult =
                createCopyResult(newOutType, newInput, subViewOp.result(), rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

mlir::LogicalResult ConcatRewrite::matchAndRewrite(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Concat Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");

    auto newOutType = typeConverter->convertType(origOp.getResult().getType());

    SmallVector<mlir::Value> allocatedBufs;

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    const auto results = origOp.per_axisAttr() ? rewriteWithAxis(origOp, newArgs, outputBuffers, rewriter)
                                               : rewriteWithOffsets(origOp, newArgs, outputBuffers, rewriter);

    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, newOutType, results, outputBuffers[0]);
    return mlir::success();
}

//
// PermuteCastRewrite
//

class PermuteCastRewrite final : public mlir::OpConversionPattern<VPU::PermuteCastOp> {
public:
    PermuteCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::PermuteCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("PermuteCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::PermuteCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PermuteCastRewrite::matchAndRewrite(VPU::PermuteCastOp origOp, OpAdaptor newArgs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found PermuteCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<VPUIP::PermuteCastOp>(origOp, newOutType, newArgs.input(), origOp.dst_orderAttr(),
                                                      origOp.mem_permAttr());
    return mlir::success();
}

//
// QuantizeCastRewriter
//

class QuantizeCastRewriter final : public mlir::OpConversionPattern<VPU::QuantizeCastOp> {
public:
    QuantizeCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::QuantizeCastOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("QuantizeCastRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::QuantizeCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeCastRewriter::matchAndRewrite(VPU::QuantizeCastOp origOp, OpAdaptor newArgs,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found QuantizeCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp.getType();

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    rewriter.replaceOpWithNewOp<VPUIP::QuantizeCastOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

//
// DistributedCastRewriter
//

class DistributedCastRewriter final : public mlir::OpConversionPattern<VPU::DistributedCastOp> {
public:
    DistributedCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::DistributedCastOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("DistributedCastRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DistributedCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DistributedCastRewriter::matchAndRewrite(VPU::DistributedCastOp origOp, OpAdaptor newArgs,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found DistributedCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<VPUIP::DistributedCastOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

//
// StubRewrite
//

class StubRewrite final : public mlir::OpConversionPattern<VPU::StubOp> {
public:
    StubRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::StubOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("StubRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::StubOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StubRewrite::matchAndRewrite(VPU::StubOp origOp, OpAdaptor newArgs,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Stub Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    SmallVector<mlir::Type> outputTypes;
    for (auto out : origOp.getResults()) {
        outputTypes.push_back(typeConverter->convertType(out.getType()));
    }

    rewriter.replaceOpWithNewOp<VPUIP::StubOp>(origOp, outputTypes, newArgs.getOperands());

    return mlir::success();
}

//
// GroupSparseTensorRewriter
//

class GroupSparseTensorRewriter final : public mlir::OpConversionPattern<VPU::GroupSparseTensorOp> {
public:
    GroupSparseTensorRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::GroupSparseTensorOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("GroupSparseTensorRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GroupSparseTensorOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupSparseTensorRewriter::matchAndRewrite(VPU::GroupSparseTensorOp origOp, OpAdaptor newArgs,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found GroupSparseTensorOp Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUIP::CompressionSchemeAttr compressionScheme = nullptr;
    if (origOp.compression_schemeAttr() != nullptr) {
        auto origCompression = origOp.compression_schemeAttr();
        compressionScheme =
                VPUIP::CompressionSchemeAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                  origCompression.getNumElems(), origCompression.getAlignment());
    }

    rewriter.replaceOpWithNewOp<VPUIP::GroupSparseBufferOp>(origOp, newArgs.data(), newArgs.sparsityMap(),
                                                            newArgs.storageElementTable(), origOp.is_weightsAttr(),
                                                            compressionScheme, origOp.seAttr().value_or(nullptr));

    return mlir::success();
}

//
// StorageElementTableRewriter
//

class StorageElementTableRewriter final : public mlir::OpConversionPattern<VPU::StorageElementTableOp> {
public:
    StorageElementTableRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::StorageElementTableOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("StorageElementTableRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::StorageElementTableOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StorageElementTableRewriter::matchAndRewrite(VPU::StorageElementTableOp origOp,
                                                                 OpAdaptor /*newArgs*/,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found StorageElementTableOp Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::StorageElementTableOp>(
            origOp, origOp.dataShapeAttr(), origOp.dataElemTypeAttr(), origOp.seSizeAttr(), origOp.seDepthAttr(),
            origOp.seAttrAttr(), origOp.dataStridesAttr(), origOp.basePtrsAttr());

    return mlir::success();
}

//
// ShapeCastRewrite
//

class ShapeCastRewrite final : public mlir::OpConversionPattern<VPU::ShapeCastOp> {
public:
    ShapeCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ShapeCastOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ShapeCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ShapeCastRewrite::matchAndRewrite(VPU::ShapeCastOp origOp, OpAdaptor newArgs,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found ShapeCast Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::ShapeCastOp>(origOp, newArgs.source(), newArgs.shape());

    _log.trace("Replaced with 'VPUIP.ShapeCastOp'");

    return mlir::success();
}

//
// LayoutCastRewrite
//

class LayoutCastRewrite final : public mlir::OpConversionPattern<VPU::LayoutCastOp> {
public:
    LayoutCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::LayoutCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("LayoutCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LayoutCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayoutCastRewrite::matchAndRewrite(VPU::LayoutCastOp origOp, OpAdaptor newArgs,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LayoutCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    const auto outOrder = DimsOrder::fromValue(origOp.output());
    const auto outMap = outOrder.toAffineMap(origOp.getContext());
    const auto mapAttr = mlir::AffineMapAttr::get(outMap);
    rewriter.replaceOpWithNewOp<VPUIP::PermuteCastOp>(origOp, newOutType, newArgs.input(), origOp.dst_orderAttr(),
                                                      mapAttr);
    return mlir::success();
}

//
// WorkloadCastRewrite
//

class WorkloadCastRewrite final : public mlir::OpConversionPattern<VPU::WorkloadCastOp> {
public:
    WorkloadCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::WorkloadCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("WorkloadCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::WorkloadCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult WorkloadCastRewrite::matchAndRewrite(VPU::WorkloadCastOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found WorkloadCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());
    rewriter.replaceOpWithNewOp<VPUIP::WorkloadCastOp>(origOp, newOutType, newArgs.input());

    return mlir::success();
}

//
// UpsamplingRewrite
//

class UpsamplingRewrite final : public mlir::OpConversionPattern<VPU::UpsamplingOp> {
public:
    UpsamplingRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::UpsamplingOp>(typeConverter, ctx), _log(log) {
        setDebugName("UpsamplingRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::UpsamplingOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UpsamplingRewrite::matchAndRewrite(VPU::UpsamplingOp origOp, OpAdaptor newArgs,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Upsampling Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());
    rewriter.replaceOpWithNewOp<VPUIP::UpsamplingUPAOp>(origOp, newArgs.input(), outputBuffers[0],
                                                        origOp.upsampling_factorAttr(), origOp.padAttr());

    return mlir::success();
}

//
// ConvertLayers2VPUIPPass
//

class ConvertLayers2VPUIPPass final : public ConvertLayers2VPUIPBase<ConvertLayers2VPUIPPass> {
public:
    explicit ConvertLayers2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    vpux::BufferizeTypeConverter typeConverter;

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<Const::ConstDialect>(isLegalOp);
    target.addIllegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp,
                      VPU::NCEEltwiseOp, VPU::NCEPermuteQuantizeOp, VPU::NCECompressConvolutionOp,
                      VPU::NCEInterpolateOp>();
    target.addLegalOp<VPU::DPUWorkloadOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp, VPU::YieldOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopyRewrite>(typeConverter, &ctx, _log);
    patterns.add<ExpandRewrite>(typeConverter, &ctx, _log);
    patterns.add<StridedSliceRewrite>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::AffineReshapeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::ReshapeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::SqueezeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::UnsqueezeOp>>(typeConverter, &ctx, _log);
    patterns.add<SliceRewrite>(typeConverter, &ctx, _log);
    patterns.add<SplitRewrite>(typeConverter, &ctx, _log);
    patterns.add<ConcatRewrite>(typeConverter, &ctx, _log);
    patterns.add<PermuteCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<QuantizeCastRewriter>(typeConverter, &ctx, _log);
    patterns.add<DistributedCastRewriter>(typeConverter, &ctx, _log);
    patterns.add<StubRewrite>(typeConverter, &ctx, _log);
    patterns.add<GroupSparseTensorRewriter>(typeConverter, &ctx, _log);
    patterns.add<StorageElementTableRewriter>(typeConverter, &ctx, _log);
    patterns.add<ShapeCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<LayoutCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<WorkloadCastRewrite>(typeConverter, &ctx, _log);
    // Track [E#81808]: implement upsampling as sw kernel
    patterns.add<UpsamplingRewrite>(typeConverter, &ctx, _log);
    Const::ConstDialect::populateBufferizePatterns(patterns, typeConverter, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUIPPass>(log);
}

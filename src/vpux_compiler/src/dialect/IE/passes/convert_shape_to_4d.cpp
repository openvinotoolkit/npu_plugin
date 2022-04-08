//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

constexpr int64_t TARGET_TENSOR_DIM = 4;

//
// ConvertShapeTo4DPass
//

class ConvertShapeTo4DPass final : public IE::ConvertShapeTo4DBase<ConvertShapeTo4DPass> {
public:
    explicit ConvertShapeTo4DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// GenericConverter
//

mlir::LogicalResult convertGeneric(mlir::Operation* origOp, mlir::ValueRange operands,
                                   mlir::ConversionPatternRewriter& rewriter, mlir::TypeConverter& typeConverter,
                                   Logger log) {
    log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(typeConverter.convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

template <class ConcreteOp>
class GenericConverter final : public mlir::OpConversionPattern<ConcreteOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<ConcreteOp>::OpAdaptor;

public:
    GenericConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ConcreteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto* typeConverter = this->getTypeConverter();
        VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

        return convertGeneric(origOp, newArgs.getOperands(), rewriter, *typeConverter, _log);
    }

private:
    Logger _log;
};

//
// FakeQuantizeConverter
//

class FakeQuantizeConverter final : public mlir::OpConversionPattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::FakeQuantizeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

struct ExtendedShape final {
    SmallVector<int64_t> shape;
    Optional<int64_t> quantizeAxis;
};

ExtendedShape extendInputShapeTo4D(IE::FakeQuantizeOp origOp) {
    // If present, axis will be pointing to the channel dimension
    const auto axis = IE::getFQAxisIndex(origOp);
    const auto inShape = origOp.input().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto perAxisCase = [](const auto& inShape, const auto& axis) -> SmallVector<int64_t> {
        // We are trying to place inShape[*axis] dimension to the outShape[1]
        switch (inShape.size()) {
        case 1: {
            return {1, inShape[0], 1, 1};
        }
        case 2: {
            if (*axis == 0) {
                return {1, inShape[0], inShape[1], 1};
            }
            return {inShape[0], inShape[1], 1, 1};
        }
        case 3: {
            if (*axis == 0) {
                return {1, inShape[0], inShape[1], inShape[2]};
            } else if (*axis == 1) {
                return {inShape[0], inShape[1], 1, inShape[2]};
            }
            VPUX_THROW("FakeQuantize 3D case doesn't support axis = {0}", *axis);
        }
        }
        VPUX_THROW("Failed to handle FakeQuantize");
    };

    const auto perTensorCase = [&](const auto& inShape) -> SmallVector<int64_t> {
        auto outShape = to_small_vector(inShape);
        outShape.insert(outShape.begin(), 4 - outShape.size(), 1);
        return outShape;
    };

    const auto outputShape = axis.hasValue() ? perAxisCase(inShape, axis) : perTensorCase(inShape);

    return {outputShape, axis};
}

mlir::Value reshapeConstInput(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value origInput,
                              llvm::Optional<int64_t> axis) {
    const auto inShape = getShape(origInput);
    if (inShape.empty()) {
        return origInput;
    }

    Shape constInputShape{1, 1, 1, 1};
    if (axis.hasValue()) {
        const auto C = Dim(1);
        constInputShape[C] = inShape[Dim(*axis)];
    }

    const auto constInputShapeAttr = getIntArrayAttr(rewriter.getContext(), constInputShape);

    return rewriter.createOrFold<IE::ReshapeOp>(loc, origInput, nullptr, false, constInputShapeAttr);
}

mlir::LogicalResult FakeQuantizeConverter::matchAndRewrite(IE::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto input4D = extendInputShapeTo4D(origOp);

    const auto inputLow = reshapeConstInput(rewriter, origOp->getLoc(), origOp.input_low(), input4D.quantizeAxis);
    const auto inputHigh = reshapeConstInput(rewriter, origOp->getLoc(), origOp.input_high(), input4D.quantizeAxis);
    const auto outputLow = reshapeConstInput(rewriter, origOp->getLoc(), origOp.output_low(), input4D.quantizeAxis);
    const auto outputHigh = reshapeConstInput(rewriter, origOp->getLoc(), origOp.output_high(), input4D.quantizeAxis);

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), input4D.shape);
    auto inputReshape =
            rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), newArgs.input(), nullptr, false, newInputShapeAttr);

    auto newFakeQuantizeOp =
            rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), inputReshape, inputLow, inputHigh, outputLow,
                                                outputHigh, origOp.levels(), origOp.auto_broadcast());

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.output()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newFakeQuantizeOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("[{0}] Replaced with 'IE::FakeQuantize'", getDebugName());

    return mlir::success();
}

//
// EltwiseOpConverter
//

template <typename EltwiseOp>
class EltwiseOpConverter final : public mlir::OpConversionPattern<EltwiseOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<EltwiseOp>::OpAdaptor;

public:
    EltwiseOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<EltwiseOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename EltwiseOp>
mlir::LogicalResult EltwiseOpConverter<EltwiseOp>::matchAndRewrite(EltwiseOp origOp, OpAdaptor,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    const auto shapeExtendAsDepth = [](vpux::NDTypeInterface inType, bool isSameInShape, bool isBatchLargeOne) {
        // There are two kinds of optimization for EltwiseOp
        // Scenario 1: Convert BxNxM to 1xBxNxM can avoid unroll at batch
        // Scenario 2: Convert 1x1xN to 1xNx1x1 can avoid expand
        if (isBatchLargeOne && inType.getRank() == 3) {
            SmallVector<int64_t> newShape(TARGET_TENSOR_DIM - inType.getRank(), 1);
            newShape.append(inType.getShape().begin(), inType.getShape().end());
            return inType.changeShape(ShapeRef(newShape));
        } else if (isSameInShape) {
            const auto greaterThanOne = [](auto dim) {
                return dim > 1;
            };

            const auto inShape = inType.getShape();
            const auto axisCount = llvm::count_if(inShape, greaterThanOne);
            if (axisCount == 1) {
                auto axis = llvm::find_if(inShape, greaterThanOne);
                VPUX_THROW_UNLESS(axis != inShape.end(), "Can not get right Axis");
                return inType.changeShape(ShapeRef({1, *axis, 1, 1}));
            }
        }

        return inType;
    };

    const auto inputReshape = [&](mlir::Value input, bool isSameInShape, bool isBatchLargeOne) {
        auto inType = input.getType().template cast<vpux::NDTypeInterface>();

        const auto in4DType =
                this->getTypeConverter()->convertType(shapeExtendAsDepth(inType, isSameInShape, isBatchLargeOne));
        const auto inShapeAttr =
                getIntArrayAttr(this->getContext(), in4DType.template cast<vpux::NDTypeInterface>().getShape());
        return rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), input, nullptr, false, inShapeAttr);
    };

    const auto shapeOne = origOp.input1().getType().template cast<vpux::NDTypeInterface>().getShape();
    const auto shapeTwo = origOp.input2().getType().template cast<vpux::NDTypeInterface>().getShape();
    bool isSameInShape = (shapeOne == shapeTwo);
    bool isBatchLargeOne = (shapeOne[Dim(0)] > 1 || shapeTwo[Dim(0)] > 1);

    const auto newIn1 = inputReshape(origOp.input1(), isSameInShape, isBatchLargeOne);
    const auto newIn2 = inputReshape(origOp.input2(), isSameInShape, isBatchLargeOne);
    const auto newIn1Shape = newIn1.getType().template cast<vpux::NDTypeInterface>().getShape();
    const auto newIn2Shape = newIn2.getType().template cast<vpux::NDTypeInterface>().getShape();

    const auto newOutShape =
            IE::broadcastEltwiseShape(newIn1Shape.raw(), newIn2Shape.raw(), origOp.auto_broadcast(), origOp->getLoc());
    VPUX_THROW_UNLESS(mlir::succeeded(newOutShape), "{0} failed to infer output shape", origOp->getLoc());
    const auto newOutType = origOp.output().getType().template cast<vpux::NDTypeInterface>().changeShape(
            ShapeRef(newOutShape.getValue()));

    auto newOp = rewriter.create<EltwiseOp>(origOp->getLoc(), newOutType, newIn1, newIn2, origOp.auto_broadcast(),
                                            origOp.post_opAttr());

    const auto outputShapeAttr = getIntArrayAttr(this->getContext(), getShape(origOp.output()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newOp.output(), nullptr, false, outputShapeAttr);

    return mlir::success();
}

//
// TopKOpConverter
//

class TopKOpConverter final : public mlir::OpConversionPattern<IE::TopKOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<IE::TopKOp>::OpAdaptor;

public:
    TopKOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::TopKOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKOpConverter::matchAndRewrite(IE::TopKOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    // The table below originates in the typeConverter behavior.
    // 1D: {M}             -> {M, 1, 1, 1}
    // 2D: {M, N}          -> {1, M, 1, N}
    // 3D: {M, N, P}       -> {M, N, 1, P}
    // 5D: {1, M, N, P, Q} -> {M, N, P, Q}
    //
    // So have the axis index mapping per shape rank
    // 1D: 0             -> 0
    // 2D: 0, 1          -> 1, 3
    // 3D: 0, 1, 2       -> 0, 1, 3
    // 5D: 0, 1, 2, 3, 4 -> -1, 0, 1, 2, 3
    //
    std::map<int64_t, SmallVector<int64_t>> rankToAxis = {{1, {0}}, {2, {1, 3}}, {3, {0, 1, 3}}, {5, {-1, 0, 1, 2, 3}}};

    const auto origInType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const int64_t origInRank = origInType.getRank();
    int64_t axis = origOp.axis();
    if (axis < 0) {
        axis += origInRank;
    }

    // Deduce the new TopK aix from map table
    const int64_t newAxis = rankToAxis[origInRank][axis];
    VPUX_THROW_WHEN(newAxis < 0, "This type of 5D TopK {0} is not supported", origInType);
    const auto newAxisAttr = getIntAttr(origOp->getContext(), newAxis);

    const auto newInType = this->getTypeConverter()->convertType(origInType);
    const auto newInShapeAttr = getIntArrayAttr(this->getContext(), newInType.cast<vpux::NDTypeInterface>().getShape());
    const auto newInReshape =
            rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, newInShapeAttr);

    auto newTopKOp = rewriter.create<IE::TopKOp>(origOp->getLoc(), newInReshape, origOp.k(), newAxisAttr,
                                                 origOp.modeAttr(), origOp.sortAttr(), origOp.element_typeAttr());

    for (auto indexResult : origOp->getResults() | indexed) {
        auto idx = checked_cast<unsigned>(indexResult.index());
        const auto origResult = indexResult.value();
        const auto outputShapeAttr = getIntArrayAttr(this->getContext(), getShape(origResult));
        const auto newOutputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), newTopKOp->getResult(idx),
                                                                           nullptr, false, outputShapeAttr);
        origResult.replaceAllUsesWith(newOutputReshape);
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// StridedSliceConverter
//

class StridedSliceConverter final : public mlir::OpConversionPattern<IE::StridedSliceOp> {
public:
    StridedSliceConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::StridedSliceOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StridedSliceConverter::matchAndRewrite(IE::StridedSliceOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::StridedSliceOp Operation '{1}'", getDebugName(), origOp->getLoc());

    SmallVector<int64_t> newInputShape;
    const auto origType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    newInputShape.append(origType.getShape().begin(), origType.getShape().end());
    for (int64_t i = 0; i < TARGET_TENSOR_DIM - origType.getRank(); ++i) {
        newInputShape.insert(newInputShape.end(), 1);
    }

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);
    auto inputReshape =
            rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), newArgs.input(), nullptr, false, newInputShapeAttr);

    auto newStridedSliceOp = rewriter.create<IE::StridedSliceOp>(
            origOp->getLoc(), inputReshape, origOp.begins(), origOp.ends(), origOp.strides(), origOp.begins_attrAttr(),
            origOp.ends_attrAttr(), origOp.strides_attrAttr(), origOp.begin_maskAttr(), origOp.end_maskAttr(),
            origOp.new_axis_maskAttr(), origOp.shrink_axis_maskAttr(), origOp.ellipsis_maskAttr());

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.output()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newStridedSliceOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("[{0}] Replaced with 'IE::StridedSlice'", getDebugName());

    return mlir::success();
}

//
// TransposeConverter
//

class TransposeConverter final : public mlir::OpConversionPattern<IE::TransposeOp> {
public:
    TransposeConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::TransposeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TransposeConverter::matchAndRewrite(IE::TransposeOp origOp, OpAdaptor,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::Transpose Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto origOrder = DimsOrder::fromAffineMap(origOp.order_value().getValue());
    const auto origPermVec = origOrder.toPermutation();

    const auto origType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = origType.getShape().raw();

    SmallVector<uint32_t> permVec;
    int32_t addDims = (int32_t)(TARGET_TENSOR_DIM - origType.getRank());

    for (auto d : origPermVec) {
        permVec.push_back(static_cast<uint32_t>(d.ind() + addDims));
    }

    for (auto i = 0; i < addDims; i++) {
        permVec.insert(permVec.begin(), addDims - i - 1);
    }

    SmallVector<int64_t> newInputShape(addDims, 1);

    for (auto i = 0; i < origType.getRank(); i++) {
        newInputShape.push_back(inputShape[i]);
    }

    const auto newOrderAffineMap = mlir::AffineMap::getPermutationMap(makeArrayRef(permVec), rewriter.getContext());

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);

    auto inputReshape =
            rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, newInputShapeAttr);

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(origOp->getLoc(), inputReshape, nullptr,
                                                           mlir::AffineMapAttr::get(newOrderAffineMap));

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.output()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newTransposeOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("[{0}] Replaced with 'IE::Tranpose'", getDebugName());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertShapeTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    const auto reshape = [](mlir::OpBuilder& builder, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());

        const auto outShapeAttr = builder.getI64ArrayAttr(dstType.getShape());
        return builder.createOrFold<IE::ReshapeOp>(loc, inputs.front(), nullptr, false, outShapeAttr);
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface type) {
        if (type.getRank() == TARGET_TENSOR_DIM) {
            return type;
        } else if (type.getRank() > TARGET_TENSOR_DIM) {
            if (type.getRank() == 5 && type.getShape()[Dim(0)] == 1) {
                SmallVector<int64_t> newShape = {type.getShape()[Dim(1)], type.getShape()[Dim(2)],
                                                 type.getShape()[Dim(3)], type.getShape()[Dim(4)]};
                return type.changeShape(ShapeRef(newShape));
            } else {
                VPUX_THROW("Tensors with rank > 4 are not supported");
            }
        } else if (type.getRank() == 3) {
            SmallVector<int64_t> newShape = {type.getShape()[Dim(0)], type.getShape()[Dim(1)], 1,
                                             type.getShape()[Dim(2)]};
            return type.changeShape(ShapeRef(newShape));
        } else if (type.getRank() == 2) {
            SmallVector<int64_t> newShape = {1, type.getShape()[Dim(0)], 1, type.getShape()[Dim(1)]};
            return type.changeShape(ShapeRef(newShape));
        } else {
            SmallVector<int64_t> newShape(TARGET_TENSOR_DIM - type.getRank(), 1);
            newShape.append(type.getShape().begin(), type.getShape().end());
            return type.changeShape(ShapeRef(newShape));
        }
    });
    typeConverter.addSourceMaterialization(reshape);
    typeConverter.addTargetMaterialization(reshape);
    typeConverter.addArgumentMaterialization(reshape);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    const auto isLegalFqOp = [&](IE::FakeQuantizeOp op) {
        const auto inShape = op.input().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto outShape = op.output().getType().cast<vpux::NDTypeInterface>().getShape();

        VPUX_THROW_WHEN(inShape != outShape,
                        "FakeQuantize must have the same shape for input and output. Got: {0} != {1}", inShape,
                        outShape);
        VPUX_THROW_WHEN(inShape.size() > TARGET_TENSOR_DIM &&
                                !(inShape.size() == (TARGET_TENSOR_DIM + 1) && inShape[Dim(0)] == 1),
                        "Tensors with rank > 4 are not supported");

        return inShape.size() == TARGET_TENSOR_DIM;
    };

    const auto isLegalEltwiseOp = [&](mlir::Operation* op) {
        if (op->getNumOperands() < 2) {
            return true;
        }
        const auto inOneShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
        const auto inTwoShape = op->getOperand(1).getType().cast<vpux::NDTypeInterface>().getShape();
        return (inOneShape.size() == TARGET_TENSOR_DIM) && (inTwoShape.size() == TARGET_TENSOR_DIM);
    };

    const auto isLegalTopKOp = [&](IE::TopKOp op) {
        const auto inShape = op.input().getType().cast<vpux::NDTypeInterface>().getShape();
        return inShape.size() == TARGET_TENSOR_DIM;
    };

    const auto isLegalStridedSLiceOp = [&](IE::StridedSliceOp op) {
        const auto inShape = op.input().getType().cast<vpux::NDTypeInterface>().getShape();

        return inShape.size() == TARGET_TENSOR_DIM;
    };

    const auto isLegalTransposeOp = [&](IE::TransposeOp op) {
        // Support non-4D tranpose
        // dimension rank >= 5 was supported and optimized by LegalizeNDMemPermutePass.
        const auto inShape = getShape(op.input());

        return (inShape.size() >= TARGET_TENSOR_DIM) || (inShape.size() == 1);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IE::IEDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::FuncOp>();
    target.addLegalOp<mlir::ReturnOp>();
    target.addDynamicallyLegalOp<IE::ClampOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReLUOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SigmoidOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::HSwishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SwishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::CosOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SqrtOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SinhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::CoshOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AsinhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AcoshOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AtanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ExpOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GeluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::DivideOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MinimumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MaximumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::PowerOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AndOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ScaleShiftOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::NotEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>(isLegalFqOp);
    target.addDynamicallyLegalOp<IE::LessOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LessEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GreaterOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GreaterEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalNotOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalOrOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalXorOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AbsOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AtanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AsinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AcosOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::PReluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LeakyReluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::TopKOp>(isLegalTopKOp);
    target.addDynamicallyLegalOp<IE::StridedSliceOp>(isLegalStridedSLiceOp);
    target.addDynamicallyLegalOp<IE::TransposeOp>(isLegalTransposeOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter<IE::ClampOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::EluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ReLUOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SigmoidOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::HSwishOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SwishOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::TanhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SinOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::CosOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SqrtOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SinhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::CoshOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AsinhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AcoshOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AtanhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ExpOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GeluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::DivideOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MinimumOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MaximumOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::PowerOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AndOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ScaleShiftOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::EqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LessOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LessEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::NotEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GreaterOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GreaterEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalNotOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalOrOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalXorOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AbsOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AtanOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AsinOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AcosOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::PReluOp>>(typeConverter, &ctx, _log);
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch == VPU::ArchKind::VPUX37XX) {
        target.addDynamicallyLegalOp<IE::ConvertOp>(isLegalOp);
        patterns.add<GenericConverter<IE::ConvertOp>>(typeConverter, &ctx, _log);
    }
    patterns.add<GenericConverter<IE::LeakyReluOp>>(typeConverter, &ctx, _log);
    patterns.add<FakeQuantizeConverter>(typeConverter, &ctx, _log);
    patterns.add<EltwiseOpConverter<IE::AddOp>>(typeConverter, &ctx, _log);
    patterns.add<EltwiseOpConverter<IE::MultiplyOp>>(typeConverter, &ctx, _log);
    patterns.add<EltwiseOpConverter<IE::SubtractOp>>(typeConverter, &ctx, _log);
    patterns.add<TopKOpConverter>(typeConverter, &ctx, _log);
    patterns.add<StridedSliceConverter>(typeConverter, &ctx, _log);
    patterns.add<TransposeConverter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertShapeTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShapeTo4DPass(Logger log) {
    return std::make_unique<ConvertShapeTo4DPass>(log);
}

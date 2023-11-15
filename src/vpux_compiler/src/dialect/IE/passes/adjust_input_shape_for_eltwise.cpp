//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/groupconvolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <numeric>

using namespace vpux;
using namespace IE;
namespace {

//
// AdjustInputShapeForEltwisePass
//
class AdjustInputShapeForEltwisePass final : public AdjustInputShapeForEltwiseBase<AdjustInputShapeForEltwisePass> {
public:
    explicit AdjustInputShapeForEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult adjustInputShape(mlir::Operation* origOp);

private:
    void safeRunOnFunc() final;
};

//
// ExpandEltwisePattern
//

class ExpandEltwisePattern {
public:
    ExpandEltwisePattern(mlir::Operation* eltwiseOp, Logger log): _eltwiseOp(eltwiseOp), _log(log) {
    }

    bool init();

    Logger getLogger() {
        return _log;
    }

    mlir::Operation* getEltwiseOperation() {
        return _eltwiseOp;
    }

    void addExpandInput(IE::ExpandOp expand) {
        _expandInputs.insert(expand);
    }

    mlir::DenseSet<IE::ExpandOp> getExpandInputs() {
        return _expandInputs;
    }

    size_t getExpandInputsNum() {
        return _expandInputs.size();
    }

    void addSliceOutput(IE::SliceOp slice) {
        _sliceOutputs.insert(slice);
    }

    void addNonSliceOutput(mlir::Operation* op) {
        _nonSliceOutputs.insert(op);
    }

    size_t getSliceOutputsNum() {
        return _sliceOutputs.size();
    }

    void setUnExpandedShape(Shape shape) {
        _unExpandedShape = std::move(shape);
    }

    void setNewExpandedShape(Shape shape) {
        _newExpandedShape = std::move(shape);
    }

private:
    mlir::Operation* _eltwiseOp;
    mlir::DenseSet<IE::ExpandOp> _expandInputs{};
    mlir::DenseSet<Const::DeclareOp> _constInputs{};
    mlir::DenseSet<mlir::Operation*> _nonExpandInputs{};
    mlir::DenseSet<IE::SliceOp> _sliceOutputs{};
    mlir::DenseSet<mlir::Operation*> _nonSliceOutputs{};
    Shape _unExpandedShape;
    Shape _newExpandedShape;
    Logger _log;

    void checkAndCorrectGroupConv();

public:
    bool opCostReduced();
    mlir::LogicalResult rewrite();
};

bool isDimExpansionReshape(ShapeRef origShape, ShapeRef reshapeShape) {
    auto getNonOneDims = [](ShapeRef shape) {
        Shape resultShape;
        llvm::copy_if(shape, std::back_inserter(resultShape), [](int64_t elem) {
            return elem != 1;
        });
        return resultShape;
    };
    return getNonOneDims(origShape) == getNonOneDims(reshapeShape);
}

void ExpandEltwisePattern::checkAndCorrectGroupConv() {
    auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(_eltwiseOp);
    if (groupConvOp == nullptr) {
        return;
    }
    auto groupSize = groupConvOp.groupsAttr().getInt();
    if (groupSize == _newExpandedShape[Dims4D::Act::C]) {
        return;
    }
    mlir::OpBuilder builder(_eltwiseOp);
    auto ctx = builder.getContext();
    auto newGroupAttr = getIntAttr(ctx, _newExpandedShape[Dims4D::Act::C]);
    auto newGroupConvOp = builder.create<IE::GroupConvolutionOp>(
            groupConvOp->getLoc(), groupConvOp.input(), groupConvOp.filter(), groupConvOp.bias(),
            groupConvOp.stridesAttr(), groupConvOp.pads_beginAttr(), groupConvOp.pads_end(),
            groupConvOp.dilationsAttr(), newGroupAttr, groupConvOp.post_opAttr());
    groupConvOp->replaceAllUsesWith(newGroupConvOp);
    auto origOutputType = groupConvOp.getType().cast<vpux::NDTypeInterface>();
    newGroupConvOp.output().setType(origOutputType.changeShape(_newExpandedShape));
    _eltwiseOp = newGroupConvOp.getOperation();
    return;
}

/* Try to match the Expand-Eltwise patterns
    Expand     Expand
      |          |
       \        /
         Eltwise
            |
      Slice (optional)

or:

   Expand     Expand
     |          |
QuantizeCast  QuantizeCast
       \        /
         Eltwise
            |
      Slice (optional)
*/
bool ExpandEltwisePattern::init() {
    auto log = _log.nest();
    // only support eltwise ops with same input and output layouts
    auto eltwiseOutputLayout = _eltwiseOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    for (auto operand : _eltwiseOp->getOperands()) {
        if (operand.getDefiningOp() && mlir::isa<Const::DeclareOp>(operand.getDefiningOp())) {
            continue;
        }

        auto inputLayout = operand.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        if (inputLayout != eltwiseOutputLayout) {
            _log.trace("Unsupported eltwise input and output layout");
            return false;
        }
    }
    // match input expands and non-expands
    for (auto operand : _eltwiseOp->getOperands()) {
        if (auto expand = operand.getDefiningOp<IE::ExpandOp>()) {
            _expandInputs.insert(expand);
        } else if (auto quantCast = operand.getDefiningOp<IE::QuantizeCastOp>()) {
            auto prevExpand = quantCast.input().getDefiningOp<IE::ExpandOp>();
            if (prevExpand) {
                _expandInputs.insert(prevExpand);
            } else {
                _nonExpandInputs.insert(operand.getDefiningOp());
            }
        } else if (auto constDeclare = operand.getDefiningOp<Const::DeclareOp>()) {
            _constInputs.insert(constDeclare);
        } else {
            _nonExpandInputs.insert(operand.getDefiningOp());
        }
    }
    log.trace("{0} Expand input(s) and {1} Const with Expand input(s) found", _expandInputs.size(),
              _constInputs.size());
    if (_expandInputs.empty()) {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }

    // match output slices or non-slices
    for (auto user : _eltwiseOp->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            _sliceOutputs.insert(slice);
        } else {
            _nonSliceOutputs.insert(user);
        }
    }
    log.trace("{0} Slice output(s) found", _sliceOutputs.size());

    // save the original shape and generate new shape
    auto expandInputOp = *_expandInputs.begin();
    _unExpandedShape = expandInputOp.input().getType().cast<vpux::NDTypeInterface>().getShape().toValues();
    for (auto expandInput : llvm::drop_begin(_expandInputs)) {
        auto otherExpandInput = expandInput.input().getType().cast<vpux::NDTypeInterface>().getShape().toValues();
        if (otherExpandInput != _unExpandedShape) {
            log.trace("The ExpandOp's input shapes are not equal, {0} and {1} separately, not supported",
                      otherExpandInput, _unExpandedShape);
            return false;
        }
    }

    auto activationDataSize =
            std::accumulate(_unExpandedShape.begin(), _unExpandedShape.end(), int64_t(1), std::multiplies<int64_t>());
    for (auto constDeclare : _constInputs) {
        auto realDataSizeResult = getBaseContentNumElements(constDeclare);
        if (mlir::failed(realDataSizeResult) ||
            (realDataSizeResult.value() != 1 && realDataSizeResult.value() != activationDataSize)) {
            log.trace("Unsupported const input {0} at {1}", constDeclare->getName(), constDeclare->getLoc());
            return false;
        }
    }

    auto newExpandedShapeResult = getShapeCastExpandedShape(_eltwiseOp, getShape(_eltwiseOp->getOperand(0)).toValues(),
                                                            _unExpandedShape, _log.nest());
    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }
    _newExpandedShape = newExpandedShapeResult.value();
    return true;
}

bool ExpandEltwisePattern::opCostReduced() {
    // check 1: all inputs are ExpandOp
    if (_nonExpandInputs.size() > 0) {
        _log.trace("{0} input op(s) are not ExpandOp", _nonExpandInputs.size());
        return false;
    }

    // check 2: when any of the expands to reduce is u8, the newly added expand cannot be fp16
    auto quantInputExpandExist = llvm::any_of(_expandInputs, [&](IE::ExpandOp expand) {
        auto outputType = expand.output().getType().cast<vpux::NDTypeInterface>();
        return outputType.getElementType().isUnsignedInteger(8);
    });
    auto floatOutputExpandToAdd = llvm::any_of(_nonSliceOutputs, [&](mlir::Operation* op) {
        auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        return inputType.getElementType().isa<mlir::FloatType>();
    });
    if (quantInputExpandExist && floatOutputExpandToAdd) {
        _log.trace("U8 Expand to reduce but float Expand to add. Expand cost will increase");
        return false;
    }
    return true;
}

/* Rewrite the pattern from:
                                        Const filter (Const bias)
   Expand      Expand          Expand    (1 elem)    (1 elem)
      |          |                  |       |         |
       \        /                    \      |       /
         Eltwise        or               GroupConv
            |                               |
      Slice (optional)               Slice (optional)

    to:                                  Const filter (Const bias)
  ShapeCast    ShapeCast        ShapeCast (broadcast) (broadcast)
      |          |                   |        |        |
       \        /                     \       |       /
         Eltwise                          GroupConv
            |                                 |
        ShapeCast                         ShapeCast
            |                                 |
          Expand                           Expand
            |                                 |
      Slice (optional)                 Slice (optional)
 */
mlir::LogicalResult ExpandEltwisePattern::rewrite() {
    mlir::OpBuilder builder(_eltwiseOp);
    auto ctx = builder.getContext();

    _log.trace("Converting unexpanded shape {0} to new aligned shape {1}", _unExpandedShape, _newExpandedShape);
    // Replace input Expands with ShapeCasts
    for (auto expand : _expandInputs) {
        auto inputValue = expand.input();
        auto inputType = inputValue.getType().cast<vpux::NDTypeInterface>();
        builder.setInsertionPointAfter(expand);
        auto inputShapeCastOp =
                builder.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), inputType.changeShape(_newExpandedShape),
                                                inputValue, getIntArrayAttr(ctx, _newExpandedShape.raw()));
        auto getOwnerIgnoreQuantizeCast = [&](mlir::OpOperand& opOperand) -> mlir::Operation* {
            auto ownerOp = opOperand.getOwner();
            while (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(ownerOp)) {
                auto quantizeUsers = quantizeCastOp.output().getUsers();
                if (quantizeUsers.empty()) {
                    return ownerOp;
                }
                ownerOp = *quantizeUsers.begin();
            }
            return ownerOp;
        };
        expand.output().replaceUsesWithIf(inputShapeCastOp.result(), [&](mlir::OpOperand& opOperand) {
            // replace only current user uses
            return getOwnerIgnoreQuantizeCast(opOperand) == _eltwiseOp;
        });
        // propagate the shape if QuantCasts exit
        auto innerOp = *inputShapeCastOp.result().getUsers().begin();
        while (innerOp != _eltwiseOp) {
            auto innerOpResult = innerOp->getResult(0);
            auto innerOutputType = innerOpResult.getType().cast<vpux::NDTypeInterface>();
            innerOp->getResult(0).setType(innerOutputType.changeShape(_newExpandedShape));
            if (innerOp->getResult(0).getUsers().empty()) {
                break;
            }
            innerOp = *innerOp->getResult(0).getUsers().begin();
        }
    }

    for (auto constDeclare : _constInputs) {
        auto contentAttr = constDeclare.getContentAttr();
        auto baseContent = contentAttr.getBaseContent();
        auto dataShape = getShape(constDeclare.getOutput()).toValues();

        Const::ContentAttr newContentAttr;
        Shape realDataShape;
        if (auto denseBaseAttr = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(denseBaseAttr);
            realDataShape = denseBaseAttr.getType().getShape();
        } else if (auto opaqueBaseAttr = baseContent.dyn_cast<Const::OpaqueElementsAttr>()) {
            newContentAttr = Const::ContentAttr::get(opaqueBaseAttr);
            realDataShape = opaqueBaseAttr.getType().getShape();
        } else {
            VPUX_THROW("Got unsupported 'baseContent' in 'ContentAttr'");
        }

        auto newConstOutputType = constDeclare.getOutput().getType().cast<vpux::NDTypeInterface>();
        const auto realDataSizeResult = getBaseContentNumElements(constDeclare);
        if (mlir::failed(realDataSizeResult)) {
            return mlir::failure();
        }
        const auto singleValueData = realDataSizeResult.value() == 1;
        if (singleValueData) {
            if (mlir::dyn_cast<IE::GroupConvolutionOp>(_eltwiseOp)) {
                for (const auto& dim : enumerate(dataShape)) {
                    if (dim.value() > 1) {
                        newContentAttr = newContentAttr.broadcast(Dim(dim.index()), _newExpandedShape[Dims4D::Act::C]);
                        auto newConstantShape = Shape(newConstOutputType.getShape().size(), int64_t(1));
                        newConstantShape[Dim(dim.index())] = _newExpandedShape[Dims4D::Act::C];
                        newConstOutputType = newConstOutputType.changeShape(newConstantShape);
                        newContentAttr = newContentAttr.reshape(newConstantShape);
                    }
                }
            } else {
                // original data shape may be [1], need to reshape new shape size like [1, 1, 1, 1], then broadcast to
                // new shape
                auto newConstantShape = Shape(_newExpandedShape.size(), int64_t(1));
                if (realDataShape.size() != _newExpandedShape.size()) {
                    newContentAttr = newContentAttr.reshape(newConstantShape);
                }
                for (const auto& dim : enumerate(_newExpandedShape)) {
                    if (dim.value() > 1) {
                        newContentAttr =
                                newContentAttr.broadcast(Dim(dim.index()), _newExpandedShape[Dim(dim.index())]);
                        newConstantShape[Dim(dim.index())] = _newExpandedShape[Dim(dim.index())];
                    }
                }
                // change newConstOutputType accordingly, which will be used to create a new const.
                if (newConstOutputType.getShape() != newConstantShape) {
                    newConstOutputType = newConstOutputType.changeShape(newConstantShape);
                }
            }
        }
        for (auto attr : contentAttr.getTransformations()) {
            if (attr.isa<Const::PadWithZeroAttr>() || attr.isa<Const::BroadcastAttr>()) {
                // Ignore
                continue;
            }
            if (attr.isa<Const::ReshapeAttr>()) {
                // Only remain the reshape attribute when it's used for dimension expansion to 4D
                // e.g., from [1x512] to [1x1x1x512]
                auto reshapeAttr = attr.cast<Const::ReshapeAttr>();
                auto reshapeShape = Shape(parseIntArrayAttr<int64_t>(reshapeAttr.getShape()));
                if (singleValueData || !isDimExpansionReshape(realDataShape, reshapeShape)) {
                    continue;
                }
            }
            newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
        }
        if (!singleValueData) {
            newContentAttr = newContentAttr.reshape(_newExpandedShape);
            newConstOutputType = newConstOutputType.changeShape(_newExpandedShape);
        }
        builder.setInsertionPoint(_eltwiseOp);
        auto newConstDeclare =
                builder.create<Const::DeclareOp>(constDeclare.getLoc(), newConstOutputType, newContentAttr);
        constDeclare.getOutput().replaceUsesWithIf(newConstDeclare.getOutput(), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() == _eltwiseOp;
        });
    }

    // Replace the eltwise GroupConv with correct attributes
    checkAndCorrectGroupConv();

    // Insert ShapeCasts and Expands after eltwise ops
    auto outputType = _eltwiseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    _eltwiseOp->getResult(0).setType(outputType.changeShape(_newExpandedShape));
    builder.setInsertionPointAfter(_eltwiseOp);
    auto outputShapeCastOp =
            builder.create<IE::ShapeCastOp>(_eltwiseOp->getLoc(), outputType.changeShape(_unExpandedShape),
                                            _eltwiseOp->getResult(0), getIntArrayAttr(ctx, _unExpandedShape.raw()));
    auto inputExpandOp = *_expandInputs.begin();
    auto newOutputExpandOp = builder.create<IE::ExpandOp>(_eltwiseOp->getLoc(), outputShapeCastOp.result(),
                                                          inputExpandOp.pads_beginAttr(), inputExpandOp.pads_endAttr());
    _eltwiseOp->getResult(0).replaceAllUsesExcept(newOutputExpandOp.output(), outputShapeCastOp);
    return mlir::success();
}

//
// ExpandEltwiseRewriter
//

template <class EltwiseOp>
class ExpandEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    ExpandEltwiseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("ExpandEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class EltwiseOp>
mlir::LogicalResult ExpandEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp layerOp,
                                                                      mlir::PatternRewriter& /*rewriter*/) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    auto pattern = ExpandEltwisePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite();
    }
    return mlir::failure();
}

//
// ExpandGroupConvRewriter
//

class ExpandGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ExpandGroupConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("ExpandGroupConvRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp layerOp,
                                                             mlir::PatternRewriter&) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());

    // Only support GroupConvolution with constant filter
    // if the GroupConvolution has bias, the bias has to be constant as well
    // the total size of filter constant is 1, as well as the bias, i.e., denseElem.getType().getNumElements() = 1
    // Kernel size must be 1x1, and must be a depthwise convolution.
    // in that case, the GroupConvolution can be considered as an Eltwise
    if (!groupConvIsEltwise(layerOp)) {
        return mlir::failure();
    }

    auto pattern = ExpandEltwisePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite();
    }
    return mlir::failure();
}

//
// ExpandPoolingPattern
//

class ExpandPoolingPattern final : public ExpandEltwisePattern {
public:
    ExpandPoolingPattern(mlir::Operation* pooling, Logger log): ExpandEltwisePattern(pooling, log) {
    }

    // Overwrite ExpandEltwisePattern::init()
    bool init();
};

/* Try to match the Expand-pooling patterns
         Expand
            |
        pooling
            |
      Slice (optional)
*/

bool ExpandPoolingPattern::init() {
    auto log = getLogger().nest();
    auto op = getEltwiseOperation();

    // match input expand
    auto operand = op->getOperand(0);
    if (auto expand = operand.getDefiningOp<IE::ExpandOp>()) {
        addExpandInput(expand);
        log.trace("{0} Expand input(s) found", getExpandInputsNum());
    } else {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }

    // match output slices or non-slices
    for (auto user : op->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            addSliceOutput(slice);
        } else {
            addNonSliceOutput(user);
        }
    }
    log.trace("{0} Slice output(s) found", getSliceOutputsNum());

    // save the original shape and generate new shape
    auto expandInputOp = *getExpandInputs().begin();
    auto unExpandedShape = expandInputOp.input().getType().cast<vpux::NDTypeInterface>().getShape().toValues();
    setUnExpandedShape(unExpandedShape);

    mlir::FailureOr<Shape> newExpandedShapeResult =
            getShapeCastExpandedShape(op, getShape(op->getOperand(0)).toValues(), unExpandedShape, log);
    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }

    auto newExpandedShape = newExpandedShapeResult.value();
    setNewExpandedShape(std::move(newExpandedShape));
    return true;
}

//
// ExpandPoolingRewriter
//

template <class PoolingOp>
class ExpandPoolingRewriter final : public mlir::OpRewritePattern<PoolingOp> {
public:
    ExpandPoolingRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<PoolingOp>(ctx), _log(log) {
        this->setDebugName("ExpandPoolingOpRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(PoolingOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class PoolingOp>
mlir::LogicalResult ExpandPoolingRewriter<PoolingOp>::matchAndRewrite(PoolingOp layerOp, mlir::PatternRewriter&) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());
    const auto supportedPooling = [](PoolingOp layerOp) {
        const auto kernels = parseIntArrayAttr<int64_t>(layerOp.kernel_size());
        const auto padStart = parseIntArrayAttr<int64_t>(layerOp.pads_begin());
        const auto padEnd = parseIntArrayAttr<int64_t>(layerOp.pads_end());
        const auto strides = parseIntArrayAttr<int64_t>(layerOp.strides());

        mlir::Value input = layerOp.input();
        mlir::Value output = layerOp.output();
        auto inputLayout = input.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        auto outputLayout = output.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        // input and output layer need to be same
        if (inputLayout != outputLayout) {
            return false;
        }

        auto hasValidKernels = llvm::all_of(kernels, [&](const auto& kernel) {
            return kernel == 1;
        });
        auto hasValidPadStart = llvm::all_of(padStart, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidPadEnd = llvm::all_of(padEnd, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidStrides = llvm::all_of(strides, [&](const auto& stride) {
            return stride == 1;
        });

        return hasValidKernels && hasValidPadStart && hasValidPadEnd && hasValidStrides;
    };
    if (!supportedPooling(layerOp)) {
        return mlir::failure();
    }

    auto pattern = ExpandPoolingPattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite();
    }

    return mlir::failure();
}

//
// ExpandPermuteQuantizePattern
//

class ExpandPermuteQuantizePattern final : public ExpandEltwisePattern {
public:
    ExpandPermuteQuantizePattern(mlir::Operation* permuteQuantize, Logger log)
            : ExpandEltwisePattern(permuteQuantize, log) {
    }

    // Overwrite ExpandEltwisePattern::init()
    bool init();

private:
    bool checkValidPermuteQuantizeOrders(IE::PermuteQuantizeOp op);
    bool checkValidPermuteQuantizePads(IE::PermuteQuantizeOp op);
    mlir::FailureOr<Shape> getWidthAlignedExpandedShape(mlir::Operation* operation, ShapeRef unExpandedShape,
                                                        Logger log);
};

//
// For PermuteQuantize shape adjustment.
// e.g
// 2x2x4 tensor:
// a11 a12 a13 a14               b11 b12 b13 b14
// a21 a22 a23 a24               b21 b22 b23 b24
// Layout in memory NCHW: a11 a12 a13 a14 a21 a22 a23 a24 b11 b12 b13 b14 b21 b22 b23 b24
// Layout in memory NHWC: a11 b11 a12 b12 a13 b13 a14 b14 a21 b21 a22 b22 a23 b23 a24 b24
//
// 2x4x2 tensor:
// a11 a12                              b11 b12
// a13 a14                              b13 b14
// a21 a22                              b21 b22
// a23 a24                              b23 b24
// Layout in memory NCHW: a11 a12 a13 a14 a21 a22 a23 a24 b11 b12 b13 b14 b21 b22 b23 b24
// Layout in memory NHWC: a11 b11 a12 b12 a13 b13 a14 b14 a21 b21 a22 b22 a23 b23 a24 b24
//
bool ExpandPermuteQuantizePattern::checkValidPermuteQuantizeOrders(IE::PermuteQuantizeOp op) {
    auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    auto inputLayout = inType.getDimsOrder();
    auto outType = op.output().getType().cast<vpux::NDTypeInterface>();
    auto outputLayout = outType.getDimsOrder();

    const auto supportedPerm = vpux::DimsOrder::NHWC.toAffineMap(op->getContext());

    return inputLayout == DimsOrder::NCHW && outputLayout == DimsOrder::NHWC && op.mem_perm() == supportedPerm;
}

bool ExpandPermuteQuantizePattern::checkValidPermuteQuantizePads(IE::PermuteQuantizeOp op) {
    const auto padStart = parseIntArrayAttr<int64_t>(op.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(op.pads_end());

    const auto nonZeroPadStart = llvm::any_of(padStart, [](auto pad) {
        return pad > 0;
    });

    const auto nonZeroPadEnd = llvm::any_of(padEnd, [](auto pad) {
        return pad > 0;
    });

    return !(nonZeroPadStart || nonZeroPadEnd);
}

mlir::FailureOr<Shape> ExpandPermuteQuantizePattern::getWidthAlignedExpandedShape(mlir::Operation* operation,
                                                                                  ShapeRef unExpandedShape,
                                                                                  Logger log) {
    auto permuteQuantize = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(operation);
    if (permuteQuantize == nullptr || !checkValidPermuteQuantizeOrders(permuteQuantize) ||
        !checkValidPermuteQuantizePads(permuteQuantize)) {
        return mlir::failure();
    }

    if (unExpandedShape.size() != 4) {
        return mlir::failure();
    }

    const auto inputType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());

    auto IH = unExpandedShape[Dims4D::Act::H];
    auto IW = unExpandedShape[Dims4D::Act::W];
    if (IH * IW % alignment != 0) {
        log.trace("Unable to adjust the input shape for op {0} at {1}, shape {2}", operation->getName(),
                  operation->getLoc(), unExpandedShape);
        return mlir::failure();
    }

    auto newExpandedShape = Shape(unExpandedShape.size(), 1);
    newExpandedShape[Dims4D::Act::N] = unExpandedShape[Dims4D::Act::N];
    newExpandedShape[Dims4D::Act::C] = unExpandedShape[Dims4D::Act::C];
    newExpandedShape[Dims4D::Act::H] = IH * IW / alignment;
    newExpandedShape[Dims4D::Act::W] = alignment;

    return newExpandedShape;
}

/* Try to match the Expand-PermuteQuantize patterns
         Expand
            |
     PermuteQuantize
            |
      Slice (optional)
*/

bool ExpandPermuteQuantizePattern::init() {
    auto log = getLogger().nest();
    auto op = getEltwiseOperation();
    auto permuteQuantize = mlir::dyn_cast<IE::PermuteQuantizeOp>(op);
    if (permuteQuantize == nullptr) {
        return false;
    }

    if (!checkValidPermuteQuantizeOrders(permuteQuantize)) {
        log.trace("Invalid PermuteQuantize layouts. '{0}' at '{1}'", permuteQuantize->getName(),
                  permuteQuantize->getLoc());
        return false;
    }

    // match input expand
    auto operand = op->getOperand(0);
    if (auto expand = operand.getDefiningOp<IE::ExpandOp>()) {
        const auto padsEnd = Shape(parseIntArrayAttr<int64_t>(expand.pads_end()));
        if (padsEnd[Dims4D::Act::N] == 0 && padsEnd[Dims4D::Act::C] == 0 && padsEnd[Dims4D::Act::H] == 0) {
            // only width expanding should be handled
            addExpandInput(expand);
        }
    }

    log.trace("{0} Expand input(s) found", getExpandInputsNum());
    if (getExpandInputsNum() == 0) {
        log.trace("Cannot find any input ExpandOp");
        return false;
    }

    // match output slices or non-slices
    for (auto user : op->getResult(0).getUsers()) {
        if (auto slice = mlir::dyn_cast<IE::SliceOp>(user)) {
            addSliceOutput(slice);
        } else {
            addNonSliceOutput(user);
        }
    }
    log.trace("{0} Slice output(s) found", getSliceOutputsNum());

    // save the original shape and generate new shape
    auto expandInputOp = *getExpandInputs().begin();
    auto unExpandedShape = expandInputOp.input().getType().cast<vpux::NDTypeInterface>().getShape().toValues();
    setUnExpandedShape(unExpandedShape);

    mlir::FailureOr<Shape> newExpandedShapeResult = getWidthAlignedExpandedShape(op, unExpandedShape, log);

    if (mlir::failed(newExpandedShapeResult)) {
        return false;
    }

    setNewExpandedShape(newExpandedShapeResult.value());
    return true;
}

//
// ExpandPermuteQuantizeRewriter
//

class ExpandPermuteQuantizeRewriter final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    ExpandPermuteQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
        setDebugName("ExpandPermuteQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandPermuteQuantizeRewriter::matchAndRewrite(IE::PermuteQuantizeOp layerOp,
                                                                   mlir::PatternRewriter&) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), layerOp->getName(), layerOp->getLoc());

    auto pattern = ExpandPermuteQuantizePattern(layerOp.getOperation(), _log);
    if (!pattern.init()) {
        return mlir::failure();
    }
    if (pattern.opCostReduced()) {
        return pattern.rewrite();
    }
    return mlir::failure();
}

void AdjustInputShapeForEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandEltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<ExpandEltwiseRewriter<IE::SubtractOp>>(&ctx, _log);
    patterns.add<ExpandEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<ExpandGroupConvRewriter>(&ctx, _log);
    patterns.add<ExpandPermuteQuantizeRewriter>(&ctx, _log);
    patterns.add<ExpandPoolingRewriter<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<ExpandPoolingRewriter<IE::MaxPoolOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}
}  // namespace

//
// createAdjustInputShapeForEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustInputShapeForEltwisePass(Logger log) {
    return std::make_unique<AdjustInputShapeForEltwisePass>(log);
}

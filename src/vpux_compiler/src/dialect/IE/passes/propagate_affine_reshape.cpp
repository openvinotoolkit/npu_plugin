//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

bool areModifiedAxesWereSplitOrMerged(SmallVector<SmallVector<int64_t>> dimMapping, ShapeRef affineInShape,
                                      ShapeRef affineOutShape, mlir::DenseSet<int64_t> modifiedAxes, Logger log) {
    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];

        for (size_t mapId = 0; mapId < mappedDim.size(); mapId++) {
            size_t outIdx = mappedDim[mapId];
            if (modifiedAxes.contains(outIdx)) {
                if (affineInShape[Dim(inIdx)] != 1 && affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)]) {
                    log.trace("Modified axis '{0}' was split or merged from several axes.", outIdx);
                    return true;
                }
            }
        }
    }

    return false;
}

//
// MoveThroughLayer
//

template <typename ConcreteOp>
class MoveThroughLayer : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughLayer(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual mlir::DenseSet<int64_t> getModifiedAxis(ConcreteOp origOp) const = 0;
    virtual SmallVector<mlir::Attribute> getNewAttrs(ConcreteOp origOp, IE::AffineReshapeOp affineReshape) const = 0;
    virtual void updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const = 0;

protected:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MoveThroughLayer<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto maybeAffineReshape = origOp.input().template getDefiningOp<IE::AffineReshapeOp>();
    if (maybeAffineReshape == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got layer: '{0}'", origOp);
    _log.trace("Parent AffineReshape: '{0}'", maybeAffineReshape);

    const auto affineInShape = getShape(maybeAffineReshape.input());
    const auto affineOutShape = getShape(maybeAffineReshape.output());

    const auto modifiedAxes = getModifiedAxis(origOp);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(maybeAffineReshape.dim_mapping());

    if (areModifiedAxesWereSplitOrMerged(dimMapping, affineInShape, affineOutShape, modifiedAxes, _log)) {
        return mlir::failure();
    }

    mlir::BlockAndValueMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {maybeAffineReshape.input()};
    mapper.map(origOp->getOperands(), makeArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*origOp.getOperation(), mapper);

    auto newAttrs = getNewAttrs(origOp, maybeAffineReshape);
    _log.trace("New attributes: '{0}'", newAttrs);

    updateAttrs(newLayerOp, newAttrs);

    vpux::inferReturnTypes(newLayerOp, vpux::InferShapedTypeMode::ALL);
    _log.trace("Create new layer: '{0}'", newLayerOp->getLoc());

    const auto outputShape = origOp.getType().getShape();
    const auto outShapeAttr = getIntArrayAttr(newLayerOp->getContext(), outputShape);

    auto newAffineReshape = rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origOp, newLayerOp->getResult(0), maybeAffineReshape.dim_mappingAttr(), outShapeAttr);
    _log.trace("Replace current layer op with new AffineReshape: '{0}'", newAffineReshape);

    return mlir::success();
}

//
// MoveThroughTranspose
//

class MoveThroughTranspose final : public MoveThroughLayer<IE::TransposeOp> {
public:
    MoveThroughTranspose(mlir::MLIRContext* ctx, Logger log): MoveThroughLayer<IE::TransposeOp>(ctx, log) {
    }

private:
    mlir::DenseSet<int64_t> getModifiedAxis(IE::TransposeOp origOp) const override;
    SmallVector<mlir::Attribute> getNewAttrs(IE::TransposeOp origOp, IE::AffineReshapeOp affineReshape) const override;
    void updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const override;
};

mlir::DenseSet<int64_t> MoveThroughTranspose::getModifiedAxis(IE::TransposeOp origOp) const {
    const auto originPerm = DimsOrder::fromAffineMap(origOp.order_value().getValue());
    const auto order = to_small_vector(irange(originPerm.numDims()) | transformed([&](uint64_t idx) {
                                           return checked_cast<uint64_t>(originPerm.dimAt(idx).ind());
                                       }));

    mlir::DenseSet<int64_t> modifiedAxes;
    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != i) {
            modifiedAxes.insert(i);
        }
    }

    return modifiedAxes;
}

SmallVector<mlir::Attribute> MoveThroughTranspose::getNewAttrs(IE::TransposeOp origOp,
                                                               IE::AffineReshapeOp affineReshape) const {
    const auto affineInShape = getShape(affineReshape.input());
    const auto affineOutShape = getShape(affineReshape.output());

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.dim_mapping());
    SmallVector<int64_t> invertedDimMapping(affineOutShape.size(), 0);

    for (size_t inDim = 0; inDim < dimMapping.size(); inDim++) {
        auto dimsArr = dimMapping[inDim];
        for (size_t i = 0; i < dimsArr.size(); i++) {
            auto outDim = dimsArr[i];
            if (affineInShape[Dim(inDim)] == affineOutShape[Dim(outDim)]) {
                invertedDimMapping[dimsArr[i]] = inDim;
                break;
            }
        }
    }

    SmallVector<unsigned> newPerm(affineInShape.size(), 0);
    const auto originPerm = DimsOrder::fromAffineMap(origOp.order_value().getValue());
    const auto order = to_small_vector(irange(originPerm.numDims()) | transformed([&](uint64_t idx) {
                                           return checked_cast<uint64_t>(originPerm.dimAt(idx).ind());
                                       }));

    for (size_t i = 0; i < newPerm.size(); i++) {
        newPerm[i] = i;
    }

    for (size_t outDim = 0; outDim < order.size(); outDim++) {
        if (order[outDim] != outDim) {
            auto inDimIdx = invertedDimMapping[outDim];
            newPerm[inDimIdx] = invertedDimMapping[order[outDim]];
        }
    }

    const auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(newPerm, origOp->getContext()));
    return SmallVector<mlir::Attribute>{orderAttr};
}

void MoveThroughTranspose::updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const {
    origOp->setAttr("order_value", newAttrs[0]);
}

//
// MoveThroughExpand
//

class MoveThroughExpand final : public MoveThroughLayer<IE::ExpandOp> {
public:
    MoveThroughExpand(mlir::MLIRContext* ctx, Logger log): MoveThroughLayer<IE::ExpandOp>(ctx, log) {
    }

private:
    SmallVector<mlir::Attribute> getNewAttrs(IE::ExpandOp origOp, IE::AffineReshapeOp affineReshape) const override;
    mlir::DenseSet<int64_t> getModifiedAxis(IE::ExpandOp origOp) const override;
    void updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const override;
};

mlir::DenseSet<int64_t> MoveThroughExpand::getModifiedAxis(IE::ExpandOp origOp) const {
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());

    mlir::DenseSet<int64_t> modifiedAxes;
    for (size_t i = 0; i < padsBegin.size(); i++) {
        if (padsBegin[i] != 0 || padsEnd[i] != 0) {
            modifiedAxes.insert(i);
        }
    }

    return modifiedAxes;
}

SmallVector<mlir::Attribute> MoveThroughExpand::getNewAttrs(IE::ExpandOp origOp,
                                                            IE::AffineReshapeOp affineReshape) const {
    const auto affineInShape = getShape(affineReshape.input());
    const auto affineOutShape = getShape(affineReshape.output());

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.dim_mapping());
    SmallVector<int64_t> invertedDimMapping(affineOutShape.size(), 0);

    for (size_t inDim = 0; inDim < dimMapping.size(); inDim++) {
        auto dimsArr = dimMapping[inDim];
        for (size_t i = 0; i < dimsArr.size(); i++) {
            auto outDim = dimsArr[i];
            if (affineInShape[Dim(inDim)] == affineOutShape[Dim(outDim)]) {
                invertedDimMapping[dimsArr[i]] = inDim;
                break;
            }
        }
    }

    SmallVector<int64_t> newPadsBegin(affineInShape.size(), 0);
    SmallVector<int64_t> newPadsEnd(affineInShape.size(), 0);

    auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());

    for (size_t outDim = 0; outDim < padsBegin.size(); outDim++) {
        auto inDimIdx = invertedDimMapping[outDim];
        if (padsBegin[outDim] != 0) {
            newPadsBegin[inDimIdx] = padsBegin[outDim];
        }
        if (padsEnd[outDim] != 0) {
            newPadsEnd[inDimIdx] = padsEnd[outDim];
        }
    }

    mlir::Builder builder(origOp->getContext());
    auto newBeginPadsAttr = builder.getI64ArrayAttr(newPadsBegin);
    auto newEndPadsAttr = builder.getI64ArrayAttr(newPadsEnd);

    return SmallVector<mlir::Attribute>{newBeginPadsAttr, newEndPadsAttr};
}

void MoveThroughExpand::updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const {
    origOp->setAttr("pads_begin", newAttrs[0]);
    origOp->setAttr("pads_end", newAttrs[1]);
}

//
// MoveThroughConcat
//

class MoveThroughConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveThroughConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::DenseSet<int64_t> getModifiedAxis(IE::ConcatOp origOp) const;
    Logger _log;
};

void getConcatAxisParameters(IE::ConcatAttrs oldAxisAttr, IE::ConcatAttrs& newAxis, mlir::ArrayAttr dimsMappingAttr,
                             mlir::Value oldInput) {
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimsMappingAttr);
    const auto outputReshapeShape = getShape(oldInput).raw();

    const auto oldAxis = oldAxisAttr.axis().getInt();
    const auto oldAxisSize = outputReshapeShape[oldAxis];

    mlir::Optional<int64_t> axis;

    for (const auto index : irange(dimMapping.size())) {
        const auto& dims = dimMapping[index];
        for (const auto& dim : dims) {
            if (outputReshapeShape[dim] == oldAxisSize) {
                auto dimIt = llvm::find_if(dims, [&](int64_t elem) {
                    return (outputReshapeShape[elem] != 1 && outputReshapeShape[elem] != oldAxisSize);
                });
                if (dimIt != dims.end()) {
                    return;
                }

                axis = index;
                break;
            }
        }

        if (axis.hasValue()) {
            break;
        }
    }

    newAxis = IE::ConcatAttrs::get(getIntAttr(dimsMappingAttr.getContext(), axis.getValue()), oldAxisAttr.offset(),
                                   oldAxisAttr.stride(), dimsMappingAttr.getContext());
}

void getConcatOffsetsParameters(mlir::ArrayAttr oldOffsets, mlir::ArrayAttr& newOffset, mlir::ArrayAttr dimsMappingAttr,
                                SmallVector<mlir::Value> oldInputs, SmallVector<mlir::Value> newValues) {
    const auto oldOffsetsList = parseIntArrayOfArrayAttr<int64_t>(oldOffsets);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimsMappingAttr);

    size_t currentIndex = 0;
    SmallVector<SmallVector<int64_t>> newOffsetsList;
    newOffsetsList.reserve(oldOffsetsList.size());

    for (auto p : zip(oldInputs, newValues)) {
        const auto inReshapeShape = getShape(std::get<1>(p)).raw();
        const auto outputReshapeShape = getShape(std::get<0>(p)).raw();

        SmallVector<int64_t> newOffset(inReshapeShape.size(), 0);
        const auto oldOffset = oldOffsetsList[currentIndex];
        int64_t prevDim = -1;
        int64_t prevOffset = -1;

        for (const auto index : irange(newOffset.size())) {
            const auto inputReshapeSize = inReshapeShape[index];

            const auto& dims = dimMapping[index];
            for (const auto& dim : dims) {
                if (inputReshapeSize == outputReshapeShape[dim]) {
                    auto dimIt = llvm::find_if(dims, [&](int64_t elem) {
                        return (outputReshapeShape[elem] != 1 && outputReshapeShape[elem] != inputReshapeSize);
                    });
                    if (dimIt != dims.end()) {
                        return;
                    }

                    newOffset[index] = oldOffset[dim];

                    // To handle the case of expanding to multiple 1, and concat on this dimension
                    // eg: 2 x ([1] -> [1, 1, 1]) -- Concat --> [1, 2, 1] {offset = [0, 0, 0], [0, 1, 0], [0, 2, 0]}
                    auto dimOneIt = llvm::find_if(dims, [&](int64_t elem) {
                        return (outputReshapeShape[elem] == 1 && oldOffset[elem] != 0);
                    });
                    if (dimOneIt != dims.end()) {
                        newOffset[index] = oldOffset[*dimOneIt];
                    }

                    if (index > 0 && newOffset[index] == prevOffset && dim == prevDim) {
                        newOffset[index] = 0;
                    } else {
                        prevOffset = newOffset[index];
                    }

                    prevDim = dim;
                    break;
                }
            }
        }

        newOffsetsList.push_back(newOffset);
        ++currentIndex;
    }

    newOffset = getIntArrayOfArray(dimsMappingAttr.getContext(), makeArrayRef(newOffsetsList));
}

mlir::DenseSet<int64_t> MoveThroughConcat::getModifiedAxis(IE::ConcatOp origOp) const {
    mlir::DenseSet<int64_t> modifiedAxes;
    if (auto perAxis = origOp.per_axisAttr()) {
        modifiedAxes.insert(perAxis.axis().getInt());
    } else {
        const auto offsets = parseIntArrayOfArrayAttr<int64_t>(origOp.static_offsetsAttr());

        for (size_t i = 0; i < offsets.size(); i++) {
            for (size_t j = 0; j < offsets[i].size(); ++j)
                if (offsets[i][j] != 0) {
                    modifiedAxes.insert(j);
                }
        }
    }

    return modifiedAxes;
}

mlir::LogicalResult MoveThroughConcat::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}]: Rewriting {1}", getDebugName(), origConcatOp->getLoc());

    auto inputs = origConcatOp.inputs();

    if (inputs.size() < 2) {
        _log.trace("[{0}]: Invalid inputs", getDebugName());
        return mlir::failure();
    }

    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(inputs.size());
    mlir::ArrayAttr dimsMapping;
    const auto modifiedAxises = getModifiedAxis(origConcatOp);

    if (modifiedAxises.empty()) {
        return mlir::failure();
    }

    for (auto input : inputs) {
        auto parentOp = input.getDefiningOp<IE::AffineReshapeOp>();

        if (parentOp == nullptr) {
            _log.trace("[{0}]: Input {1} is not AffineReshape result", getDebugName(), input.getLoc());
            return mlir::failure();
        }

        if (!newInputs.empty()) {
            auto prevInput = newInputs.back();

            if (getShape(prevInput).size() != getShape(parentOp.input()).size()) {
                _log.trace("[{0}]: Input {1} has different shape than others", getDebugName(), parentOp.getLoc());
                return mlir::failure();
            }
        }

        if (dimsMapping != nullptr) {
            if (parentOp.dim_mapping() != dimsMapping) {
                _log.trace("[{0}]: Input {1} has different mapping from others", getDebugName(), parentOp.getLoc());
                return mlir::failure();
            }
        } else {
            dimsMapping = parentOp.dim_mapping();
        }

        const auto affineInputShape = getShape(parentOp.input());
        const auto affineOutputShape = getShape(parentOp.output());

        const auto dimMappingList = parseIntArrayOfArrayAttr<int64_t>(dimsMapping);
        if (areModifiedAxesWereSplitOrMerged(dimMappingList, affineInputShape, affineOutputShape, modifiedAxises,
                                             _log.nest())) {
            return mlir::failure();
        }

        newInputs.push_back(parentOp.input());
    }

    VPUX_THROW_WHEN(dimsMapping == nullptr, "Cannot get mapping from Reshapes");

    IE::ConcatAttrs newAxis;
    mlir::ArrayAttr newOffsets;
    if (origConcatOp.per_axisAttr() != nullptr) {
        getConcatAxisParameters(origConcatOp.per_axisAttr(), newAxis, dimsMapping, inputs.front());
    } else if (origConcatOp.static_offsetsAttr() != nullptr) {
        getConcatOffsetsParameters(origConcatOp.static_offsetsAttr(), newOffsets, dimsMapping, inputs, newInputs);
    } else {
        VPUX_THROW("Incorrent Concat parameters");
    }

    if ((origConcatOp.per_axis() && newAxis == nullptr) || (!origConcatOp.per_axis() && newOffsets == nullptr)) {
        _log.trace("[{0}]: Concat parameters couldn't be calculated", getDebugName(), origConcatOp.getLoc());
        return mlir::failure();
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origConcatOp.getLoc(), newInputs, newAxis, newOffsets);

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origConcatOp, newConcat, dimsMapping,
            getIntArrayAttr(origConcatOp.getContext(), getShape(origConcatOp).raw()));

    return mlir::success();
}

//
// PropagateAffineReshape
//

class PropagateAffineReshape final : public IE::PropagateAffineReshapeBase<PropagateAffineReshape> {
public:
    explicit PropagateAffineReshape(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateAffineReshape::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughTranspose>(&ctx, _log);
    patterns.add<MoveThroughExpand>(&ctx, _log);
    patterns.add<MoveThroughConcat>(&ctx, _log);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateAffineReshapePass(Logger log) {
    return std::make_unique<PropagateAffineReshape>(log);
}

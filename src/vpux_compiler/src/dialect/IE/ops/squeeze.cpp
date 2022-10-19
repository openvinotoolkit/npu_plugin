//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getAxes
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::SqueezeOpAdaptor squeeze, mlir::Location loc) {
    if (squeeze.axes() != nullptr && squeeze.axes_value() != nullptr) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (squeeze.axes() == nullptr && squeeze.axes_value() == nullptr) {
        return errorAt(loc, "Missed axes representation");
    }

    if (squeeze.axes_value() != nullptr) {
        return parseIntArrayAttr<int64_t>(squeeze.axes_value());
    }

    auto axesConst = squeeze.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = squeeze.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    return axes;
}

//
// inferOutputLayout
//

DimsOrder inferOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axesVec, ArrayRef<int64_t> inShape) {
    SmallVector<vpux::Dim> perm;
    SmallVector<int64_t> axes = axesVec;

    // If axes attr is empty, find all dims equal to 1
    if (axes.empty()) {
        for (auto inInd : irange(inShape.size())) {
            if (inShape[inInd] == 1) {
                axes.push_back(inInd);
            }
        }
    }

    // Iterate over input dims in the given order and push back corresponding output dims.
    for (const auto& p : inPerm) {
        // Skip over squeezed dim
        if (llvm::find(axes, p.ind()) != axes.end())
            continue;

        auto dim = p.ind();
        // Decrement input dim index by the number of squeezed axes smaller than itself
        for (const auto& squeezeAxis : axes) {
            if (p.ind() > squeezeAxis) {
                dim--;
            }
        }

        perm.push_back(vpux::Dim(dim));
    }

    return DimsOrder::fromPermutation(makeArrayRef(perm));
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::SqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SqueezeOpAdaptor squeeze(operands, attrs);
    if (mlir::failed(squeeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = getAxes(squeeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = squeeze.input();
    const auto inType = input.getType().cast<mlir::RankedTensorType>();
    const auto inShape = inType.getShape();
    const auto inOrder = DimsOrder::fromValue(input);

    SmallVector<int64_t> outShape;

    if (axes->empty()) {
        for (auto dim : inShape) {
            if (dim != 1) {
                outShape.push_back(dim);
            }
        }
    } else {
        size_t axesInd = 0;
        for (auto inInd : irange(inShape.size())) {
            if (axesInd < axes->size()) {
                const auto nextAxisInd = checked_cast<size_t>(axes.getValue()[axesInd]);

                if (nextAxisInd < inInd) {
                    return errorAt(loc, "Axis '{0}' was occurred twice", nextAxisInd);
                }

                if (nextAxisInd == inInd) {
                    if (inShape[inInd] != 1) {
                        return errorAt(loc, "Can't exclude '{0}' dimension, it is not equal to 1", nextAxisInd);
                    }

                    ++axesInd;

                    continue;
                }
            }

            outShape.push_back(inShape[inInd]);
        }
    }

    const auto outDesc = IE::getTensorAttr(ctx, inferOutputLayout(inOrder.toPermutation(), axes.getValue(), inShape),
                                           IE::getMemorySpace(inType));

    inferredReturnShapes.emplace_back(makeArrayRef(outShape), inType.getElementType(), outDesc);
    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::IE::SqueezeOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto axes = parseIntArrayAttr<int64_t>(axes_value().getValue());
    const auto inShape = input().getType().cast<mlir::RankedTensorType>().getShape();
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();

    info.setInput(0, inOrder);
    info.setOutput(0, inferOutputLayout(inPermutation, axes, inShape));
}

//
// fold
//

mlir::OpFoldResult vpux::IE::SqueezeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reshape(getShape(output()));
    }

    return nullptr;
}

//
// FuseWithReshape
//

namespace {

class FuseWithReshape final : public mlir::OpRewritePattern<IE::SqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::SqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithReshape::matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::SqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.axes_value().hasValue()) {
        return mlir::failure();
    }

    const auto axes = getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.getValue());

    rewriter.replaceOpWithNewOp<IE::SqueezeOp>(origOp, origOp.input(), nullptr, axesAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SqueezeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                      mlir::MLIRContext* context) {
    patterns.insert<FuseWithReshape>(context);
    patterns.insert<ConvertConstToAttr>(context);
}

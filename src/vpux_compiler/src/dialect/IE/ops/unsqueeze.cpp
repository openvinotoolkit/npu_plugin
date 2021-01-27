//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::UnsqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::UnsqueezeOpAdaptor unsqueeze(operands, attrs);
    if (mlir::failed(unsqueeze.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = unsqueeze.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    auto axesConst = unsqueeze.axes().getDefiningOp<ConstantInterface>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for axes");
    }

    auto axes = to_small_vector(axesConst.getContent().getValues<int64_t>());
    std::sort(axes.begin(), axes.end());
    for (auto& axis : axes) {
        if (axis < 0) {
            axis = axis + checked_cast<int64_t>(inShape.size() + axes.size());
        }
    }

    SmallVector<int64_t> outShape(inShape.size() + axes.size());

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(outShape.size())) {
        if (axesInd < axes.size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

            if (nextAxisInd < outInd) {
                return errorAt(loc, "Axis '{0}' was occured twice", nextAxisInd);
            }

            if (nextAxisInd == outInd) {
                outShape[outInd] = 1;
                ++axesInd;
                continue;
            }
        }

        if (inInd < inShape.size()) {
            outShape[outInd] = inShape[inInd];
            ++inInd;
            continue;
        }
    }
    if (inInd != inShape.size() || axesInd != axes.size()) {
        return errorAt(loc, "Inconsistent parameters");
    }

    inferredReturnShapes.emplace_back(makeArrayRef(outShape), inType.getElementType());
    return mlir::success();
}

//
// UseExpansionReshape
//

namespace {

class UseExpansionReshape final : public mlir::OpRewritePattern<IE::UnsqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::UnsqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult UseExpansionReshape::matchAndRewrite(IE::UnsqueezeOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto axesConst = origOp.axes().getDefiningOp<ConstantInterface>();
    if (axesConst == nullptr) {
        return mlir::failure();
    }

    const auto inType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    auto axes = to_small_vector(axesConst.getContent().getValues<int64_t>());
    std::sort(axes.begin(), axes.end());
    for (auto& axis : axes) {
        if (axis < 0) {
            axis = axis + checked_cast<int64_t>(inShape.size() + axes.size());
        }
    }

    //
    // Use expansion linalg.tensor_reshape:
    //
    //   input tensor: (i, j)
    //   output tensor: (1, i, 1, j, 1)
    //
    // Reassotion maps (output to input):
    //   (d0, d1, d2, d3, d4) -> (d0, d1)      : shape[d0] = 1, shape[d1] = i
    //   (d0, d1, d2, d3, d4) -> (d2, d3, d4)  : shape[d2] = 1, shape[d3] = j, shape[d4] = 1
    //

    SmallVector<mlir::linalg::ReassociationIndices> indices(inShape.size());

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(inShape.size() + axes.size())) {
        if (axesInd < axes.size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

            if (nextAxisInd == outInd) {
                indices[inInd].push_back(outInd);
                ++axesInd;
                continue;
            }
        }

        indices[inInd].push_back(outInd);

        if (inInd + 1 < inShape.size()) {
            ++inInd;
        }
    }

    rewriter.replaceOpWithNewOp<mlir::linalg::TensorReshapeOp>(origOp, origOp.getType(), origOp.input(),
                                                               makeArrayRef(indices));

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::UnsqueezeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                        mlir::MLIRContext* ctx) {
    patterns.insert<UseExpansionReshape>(ctx);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::UnsqueezeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<ConstContentAttr>()) {
        return attr;
    }

    return nullptr;
}

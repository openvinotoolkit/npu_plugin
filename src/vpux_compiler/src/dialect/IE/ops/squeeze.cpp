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

mlir::LogicalResult vpux::IE::SqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SqueezeOpAdaptor squeeze(operands, attrs);
    if (mlir::failed(squeeze.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = squeeze.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    auto axesConst = squeeze.axes().getDefiningOp<ConstantInterface>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for axes");
    }

    auto axes = to_small_vector(axesConst.getContent().getValues<int64_t>());
    std::sort(axes.begin(), axes.end());
    for (auto& axis : axes) {
        if (axis < 0) {
            axis = axis + checked_cast<int64_t>(inShape.size());
        }
    }

    SmallVector<int64_t> outShape;

    if (axes.empty()) {
        for (auto dim : inShape) {
            if (dim != 1) {
                outShape.push_back(dim);
            }
        }
    } else {
        size_t axesInd = 0;
        for (auto inInd : irange(inShape.size())) {
            if (axesInd < axes.size()) {
                const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

                if (nextAxisInd < inInd) {
                    return errorAt(loc, "Axis '{0}' was occured twice", nextAxisInd);
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

    inferredReturnShapes.emplace_back(makeArrayRef(outShape), inType.getElementType());
    return mlir::success();
}

//
// UseCollapsingReshape
//

namespace {

class UseCollapsingReshape final : public mlir::OpRewritePattern<IE::SqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::SqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult UseCollapsingReshape::matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto axesConst = origOp.axes().getDefiningOp<ConstantInterface>();
    if (axesConst == nullptr) {
        return mlir::failure();
    }

    const auto inType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    const auto outType = origOp.output().getType().cast<mlir::ShapedType>();
    const auto outShape = outType.getShape();

    if (outShape.empty()) {
        // Corner case - collapse to scalar.
        return mlir::failure();
    }

    auto axes = to_small_vector(axesConst.getContent().getValues<int64_t>());
    std::sort(axes.begin(), axes.end());
    for (auto& axis : axes) {
        if (axis < 0) {
            axis = axis + checked_cast<int64_t>(inShape.size());
        }
    }

    //
    // Use collapsing linalg.tensor_reshape:
    //
    //   input tensor: (1, i, 1, j, 1)
    //   output tensor: (i, j)
    //
    // Reassotion maps (input to output):
    //   (d0, d1, d2, d3, d4) -> (d0, d1)      : shape[d0] = 1, shape[d1] = i
    //   (d0, d1, d2, d3, d4) -> (d2, d3, d4)  : shape[d2] = 1, shape[d3] = j, shape[d4] = 1
    //

    SmallVector<mlir::linalg::ReassociationIndices> indices(outShape.size());

    if (axes.empty()) {
        size_t outInd = 0;
        for (auto inInd : irange(inShape.size())) {
            if (inShape[inInd] == 1) {
                indices[outInd].push_back(inInd);
                continue;
            }

            indices[outInd].push_back(inInd);

            if (outInd + 1 < outShape.size()) {
                ++outInd;
            }
        }
    } else {
        size_t outInd = 0;
        size_t axesInd = 0;
        for (auto inInd : irange(inShape.size())) {
            if (axesInd < axes.size()) {
                const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

                if (nextAxisInd == inInd) {
                    indices[outInd].push_back(inInd);
                    ++axesInd;
                    continue;
                }
            }

            indices[outInd].push_back(inInd);

            if (outInd + 1 < outShape.size()) {
                ++outInd;
            }
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

void vpux::IE::SqueezeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                      mlir::MLIRContext* ctx) {
    patterns.insert<UseCollapsingReshape>(ctx);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::SqueezeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<ConstContentAttr>()) {
        return attr;
    }

    return nullptr;
}

//
// ViewLikeInterface
//

mlir::Value vpux::IE::SqueezeOp::getViewSource() {
    return input();
}

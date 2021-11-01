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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

namespace {

//
// inferOutputLayout
//

mlir::FailureOr<DimsOrder> inferOutputLayout(const DimArr& inPerm, mlir::ArrayAttr dimMapAttr) {
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimMapAttr);
    SmallVector<vpux::Dim> perm;

    // Iterate over input dims in the given order and push back corresponding output dims as indicated by the op's
    // dim_mapping. The result is the permutation of output dims.
    bool layoutInferFail = false;
    for (auto pIt = inPerm.begin(); pIt != inPerm.end(); ++pIt) {
        const auto outputDims = dimMapping[pIt->ind()];
        for (const auto& dim : outputDims) {
            const auto outDim = vpux::Dim(dim);

            // Ensure input dim order is not switched.
            // E.g. nchw -> c'h'w', with n = c', c = h', h * w = w'
            // Layouts 0123 and 0132 would both produce 012 output layout, but
            // the content of w' would not be the same.
            if (!perm.empty() && perm.back() == outDim) {
                layoutInferFail = std::prev(pIt)->ind() > pIt->ind();
                if (layoutInferFail == true) {
                    return mlir::failure();
                }

                continue;
            }

            perm.push_back(outDim);
        }
    }

    // Check that the resulting output permutation does not have duplicate dims
    SmallVector<vpux::Dim> temp(perm);
    llvm::sort(temp.begin(), temp.end(), [](const vpux::Dim& dim0, const vpux::Dim& dim1) {
        return dim0.ind() < dim1.ind();
    });

    if (std::adjacent_find(temp.begin(), temp.end()) != temp.end())
        return mlir::failure();

    return DimsOrder::fromPermutation(makeArrayRef(perm));
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::AffineReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AffineReshapeOpAdaptor affineReshape(operands, attrs);
    if (mlir::failed(affineReshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(affineReshape.shape_value());
    const auto input = affineReshape.input();
    const auto inType = input.getType().cast<mlir::RankedTensorType>();
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outputLayout = inferOutputLayout(inOrder.toPermutation(), affineReshape.dim_mapping());
    if (mlir::failed(outputLayout)) {
        return mlir::failure();
    }

    const auto outDesc = IE::getTensorAttr(ctx, outputLayout.getValue(), IE::getMemorySpace(inType));
    const auto newType = changeShape(inType, ShapeRef(outShape));

    inferredReturnShapes.emplace_back(outShape, newType.getElementType(), outDesc);
    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::IE::AffineReshapeOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();
    const auto outPermutation = inferOutputLayout(inPermutation, dim_mapping());
    if (mlir::failed(outPermutation)) {
        IE::fillDefaultLayoutInfo(info);
        return;
    }

    info.setInput(0, inOrder);
    info.setOutput(0, outPermutation.getValue());
}

//
// fold
//

mlir::OpFoldResult vpux::IE::AffineReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
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
class FuseWithReshape final : public mlir::OpRewritePattern<IE::AffineReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::AffineReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AffineReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithReshape::matchAndRewrite(IE::AffineReshapeOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // Fusing AffineReshape with any of the above mentioned ops might result in another AffineReshape or not,
    // depending on the resulting input and output shapes.
    // E. g. 1 x 24 x 2 x 2 -> AffineReshape -> 1 x 24 x 4 -> AffineReshape -> 1 x 24 x 4 x 1
    //       mapping: id0 = od0, id1 = od1 and id2 * id3 = od2 * od3 (not an AffineReshape)
    // If the Reshape that replaces the two ops ends up being a valid AffineReshape, then it will be converted by
    // Reshape's canonicalizer.
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::AffineReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.insert<FuseWithReshape>(ctx);
}

//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// inferReturnTypeComponents
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getReshapeOutputShape(mlir::Location loc, IE::ReshapeOpAdaptor reshape) {
    auto shapeConst = reshape.shape().getDefiningOp<ConstantInterface>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    auto shapeVec = to_small_vector(shapeConst.getContent().getValues<int64_t>());

    const auto specialZero = reshape.special_zero();

    const auto zeroDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == 0;
    });
    const auto negativeDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == -1;
    });

    if (negativeDims > 1) {
        return errorAt(loc, "Shape can not contain more than 1 negative value");
    }

    if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
        return shapeVec;
    } else {
        const auto inShape = to_small_vector(reshape.input().getType().cast<mlir::ShapedType>().getShape());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i < inShape.size()) {
                    v = inShape[i];
                } else {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }
            }

            if (v > 0) {
                if (dividend % v != 0) {
                    return errorAt(loc, "Shape value at '{0}' ('{1}') is invalid", i, v);
                }

                dividend /= v;
            }
        }

        if (negativeDims > 0) {
            const auto negIt = std::find(shapeVec.begin(), shapeVec.end(), -1);
            VPUX_THROW_UNLESS(negIt != shapeVec.end(), "Shape vector broken");

            *negIt = dividend;
        }

        return shapeVec;
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::ReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReshapeOpAdaptor reshape(operands, attrs);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getReshapeOutputShape(loc, reshape);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    inferredReturnShapes.emplace_back(outShape.getValue(),
                                      reshape.input().getType().cast<mlir::ShapedType>().getElementType());
    return mlir::success();
}

//
// UseLinalgReshape
//

namespace {

class UseLinalgReshape final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    using ReshapeSpec = SmallVector<mlir::linalg::ReassociationIndices>;

    static mlir::FailureOr<ReshapeSpec> getCollapsingSpec(IE::ReshapeOp origOp, Logger log);
    static mlir::FailureOr<ReshapeSpec> getExpandingSpec(IE::ReshapeOp origOp, Logger log);
};

mlir::FailureOr<UseLinalgReshape::ReshapeSpec> UseLinalgReshape::getCollapsingSpec(IE::ReshapeOp origOp, Logger log) {
    log.trace("Collapsing reshape");

    const auto inType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    auto shapeConst = origOp.shape().getDefiningOp<ConstantInterface>();
    const auto shapeSpec = shapeConst.getContent().getValues<int64_t>();

    ReshapeSpec indices(shapeSpec.size());

    //
    // Check all indexes in shape specification up to `-1` value.
    // Detect cases when input dimensions are used as is.
    //

    size_t outMergeInd = shapeSpec.size() - 1;
    size_t inMergeStart = shapeSpec.size();
    for (auto specInd : irange(shapeSpec.size())) {
        const auto specVal = shapeSpec[specInd];

        log.nest(1).trace("Check spec at '{0}' : '{1}'", specInd, specVal);

        if (specVal == -1) {
            outMergeInd = specInd;
            inMergeStart = specInd;
            break;
        }

        if (specVal == 0) {
            if (!origOp.special_zero()) {
                return mlir::failure();
            }
        } else if (inShape[specInd] != specVal) {
            return mlir::failure();
        }

        log.nest(2).trace("Input index '{0}' maps to output index '{1}'", specInd, specInd);
        indices[specInd].push_back(specInd);
    }

    //
    // Check all indexes in shape specification after `-1` value.
    // Detect cases when input dimensions are used as is.
    //

    size_t inMergeEnd = inShape.size();
    for (auto specInd : irange(outMergeInd + 1, shapeSpec.size()) | reversed) {
        const auto specVal = shapeSpec[specInd];

        log.nest(1).trace("Check spec at '{0}' : '{1}'", specInd, specVal);

        if (specVal == -1) {
            return mlir::failure();
        }

        if (inMergeEnd <= inMergeStart) {
            return mlir::failure();
        }

        if (specVal == 0) {
            if (!origOp.special_zero()) {
                return mlir::failure();
            }
        } else if (inShape[inMergeEnd - 1] != specVal) {
            return mlir::failure();
        }

        log.nest(2).trace("Input index '{0}' maps to output index '{1}'", inMergeEnd - 1, specInd);

        indices[specInd].push_back(inMergeEnd - 1);
        --inMergeEnd;
    }

    //
    // Collect all input indexes, which are included into `-1` specification value.
    //

    log.nest().trace("Input indices range '[{0}, {1})'  maps to output index '{2}'", inMergeStart, inMergeEnd,
                     outMergeInd);

    for (auto inInd : irange(inMergeStart, inMergeEnd)) {
        indices[outMergeInd].push_back(inInd);
    }

    return indices;
}

mlir::FailureOr<UseLinalgReshape::ReshapeSpec> UseLinalgReshape::getExpandingSpec(IE::ReshapeOp origOp, Logger log) {
    log.trace("Expanding reshape");

    const auto inType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    auto shapeConst = origOp.shape().getDefiningOp<ConstantInterface>();
    const auto shapeSpec = shapeConst.getContent().getValues<int64_t>();

    ReshapeSpec indices(inShape.size());

    //
    // Check all indexes in shape specification up to `-1` value.
    // Detect cases when input dimensions are used as is or output shape is expanded with ones.
    //

    size_t inInd1 = 0;
    size_t outMergeStart = shapeSpec.size();
    for (auto specInd : irange(shapeSpec.size())) {
        const auto specVal = shapeSpec[specInd];

        log.nest(1).trace("Check spec at '{0}' : '{1}'", specInd, specVal);

        if (specVal == -1) {
            outMergeStart = specInd;
            break;
        }

        if (inInd1 >= inShape.size()) {
            return mlir::failure();
        }

        if (specVal == 0) {
            if (!origOp.special_zero()) {
                return mlir::failure();
            }
        } else if (specVal != 1 && inShape[inInd1] != specVal) {
            return mlir::failure();
        }

        log.nest(2).trace("Output index '{0}' maps to input index '{1}'", specInd, inInd1);
        indices[inInd1].push_back(specInd);

        if (specVal != 1) {
            ++inInd1;
        }
    }

    //
    // Check all indexes in shape specification after `-1` value.
    // Detect cases when input dimensions are used as is.
    //

    size_t outMergeEnd = shapeSpec.size();
    if (inInd1 < inShape.size()) {
        for (auto inInd2 : irange(inInd1 + 1, inShape.size()) | reversed) {
            const auto specVal = shapeSpec[outMergeEnd - 1];

            log.nest().trace("Check spec at '{0}' : '{1}'", outMergeEnd - 1, specVal);

            if (specVal == -1) {
                return mlir::failure();
            }

            if (outMergeEnd <= outMergeStart) {
                return mlir::failure();
            }

            if (specVal == 0) {
                if (!origOp.special_zero() || inInd2 <= inInd1) {
                    return mlir::failure();
                }
            } else if (specVal != 1 && inShape[inInd2] != specVal) {
                return mlir::failure();
            }

            log.nest(2).trace("Output index '{0}' maps to input index '{1}'", outMergeEnd - 1, inInd2);

            indices[inInd2].push_back(outMergeEnd - 1);
            --outMergeEnd;
        }
    }

    //
    // Collect all output indexes, which are included into `-1` specification value.
    //

    log.nest().trace("Output indices range '[{0}, {1})'  maps to input index '{2}'", outMergeStart, outMergeEnd,
                     inInd1);

    for (auto outInd : irange(outMergeStart, outMergeEnd)) {
        indices[inInd1].push_back(outInd);
    }

    return indices;
}

mlir::LogicalResult UseLinalgReshape::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto log = Logger::global().nest("IE.Reshape", 0);

    auto shapeConst = origOp.shape().getDefiningOp<ConstantInterface>();
    if (shapeConst == nullptr) {
        return mlir::failure();
    }

    const auto inType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();
    const auto shapeSpec = shapeConst.getContent().getValues<int64_t>();

    log.trace("Got IE.Reshape at '{0}' : '{1}' / '[{2}]' -> '{3}'", origOp.getLoc(), inShape, make_range(shapeSpec),
              origOp.getType().getShape());
    log = log.nest();

    auto indices = shapeSpec.size() < inShape.size() ? getCollapsingSpec(origOp, log) : getExpandingSpec(origOp, log);

    if (mlir::failed(indices)) {
        return mlir::failure();
    }

    for (auto& vec : indices.getValue()) {
        std::sort(vec.begin(), vec.end());
    }

    log.trace("Reassotiation indices : '{0}'", indices.getValue());

    rewriter.replaceOpWithNewOp<mlir::linalg::TensorReshapeOp>(origOp, origOp.getType(), origOp.input(),
                                                               makeArrayRef(indices.getValue()));

    return mlir::success();
}

}  // namespace

//
// MergeTwoReshapeOps
//

namespace {

class MergeTwoReshapeOps final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult MergeTwoReshapeOps::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    if (!mlir::isa<mlir::linalg::TensorReshapeOp, IE::ReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    auto outputShape = origOp.getType().getShape();
    const auto outShapeType = mlir::RankedTensorType::get({checked_cast<int64_t>(outputShape.size())},
                                                          getSInt64Type(origOp->getContext()));
    const auto outputShapeAttr = mlir::DenseElementsAttr::get(outShapeType, makeArrayRef(outputShape));
    auto newShape = rewriter.create<IE::ConstantOp>(origOp->getLoc(), outShapeType, outputShapeAttr);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), newShape, origOp.special_zero());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.insert<MergeTwoReshapeOps>(ctx);
    patterns.insert<UseLinalgReshape>(ctx);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
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

mlir::Value vpux::IE::ReshapeOp::getViewSource() {
    return input();
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
using namespace IE;
namespace {

//
// MovePermutePostEltwisePass
//

class MovePermutePostEltwisePass final : public MovePermutePostEltwiseBase<MovePermutePostEltwisePass> {
public:
    MovePermutePostEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// PermuteEltwiseRewriter
//

template <class EltwiseOp>
class PermuteEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    PermuteEltwiseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("PermuteEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Get the mapped Shape of the mapping from the source layout to the result + permutation layout
// with no data change
// e.g., source layout NCHW, shape is [n, c, h, w], mem_perm layout is NWCH
// return layout NHWC, the mapped result shape should be [n, h, w, c]
Shape getMappedShape(DimsOrder sourceLayout, DimsOrder resultLayout, DimsOrder memPermLayout, ShapeRef sourceShape) {
    auto resultShape = Shape(sourceShape.size(), 1);
    auto mappedShape = resultShape;
    auto sourcePerm = sourceLayout.toPermutation();
    auto resultPerm = resultLayout.toPermutation();
    auto memPerm = memPermLayout.toPermutation();
    for (auto dim : irange(sourceShape.size())) {
        resultShape[resultPerm[dim]] = sourceShape[sourcePerm[dim]];
    }
    for (auto dim : irange(mappedShape.size())) {
        mappedShape[memPerm[dim]] = resultShape[resultPerm[dim]];
    }
    return mappedShape;
}

IE::MemPermuteOp getEltwiseInputPermute(mlir::Value eltwiseInput) {
    auto parentOp = eltwiseInput.getDefiningOp();
    while (parentOp) {
        if (auto parentPermute = mlir::dyn_cast<IE::MemPermuteOp>(parentOp)) {
            // Case when there is a MemPermute and the processing of EltWise above adds another MemPermute
            // Skipping now, as canonicalizer takes care of optimizing both MemPermutes
            auto grandParentOp = parentPermute.input().getDefiningOp();
            if (grandParentOp != nullptr && mlir::isa<IE::MemPermuteOp>(grandParentOp)) {
                auto grandParentPermute = mlir::cast<IE::MemPermuteOp>(grandParentOp);
                if (parentPermute.dst_order() == grandParentPermute.dst_order()) {
                    return nullptr;
                }
            }
            return parentPermute;
        } else if (auto parentQuantizeCast = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentQuantizeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentQuantizeCast.input().getDefiningOp();
            continue;
        } else if (auto parentShapeCast = mlir::dyn_cast<IE::ShapeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentShapeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentShapeCast.source().getDefiningOp();
            continue;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

SmallVector<IE::MemPermuteOp> getPermutesToMove(ArrayRef<IE::MemPermuteOp> permutes) {
    if (permutes.size() == 1) {
        return SmallVector<IE::MemPermuteOp>({permutes[0]});
    }
    if (permutes.size() == 2) {
        if (permutes[0] == permutes[1]) {
            return SmallVector<IE::MemPermuteOp>({permutes[0]});
        } else {
            return SmallVector<IE::MemPermuteOp>({permutes[0], permutes[1]});
        }
    }
    VPUX_THROW("getPermutesToMove: Unsupported number of elements. Expected 1 or 2, got {0}", permutes.size());
}

bool isSplatConstant(Const::DeclareOp constOp) {
    if (constOp == nullptr) {
        return false;
    }
    return constOp.content().isSplat();
}

/* Rewrite the pattern from:

   Permute      Permute
      |          |
(QuantizeCast) (QuantizeCast)
       \        /
         Eltwise
            |
      (QuantizeCast)
           ...

    to:
PermuteCast  PermuteCast
      |          |
 ShapeCast   ShapeCast
      |          |
(QuantizeCast) (QuantizeCast)
       \        /
         Eltwise
            |
     (QuantizeCast)
            |
        ShapeCast
            |
       PermuteCast
            |
         Permute
            |
           ...
 */
template <class EltwiseOp>
mlir::LogicalResult PermuteEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp eltwiseOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), eltwiseOp->getName(), eltwiseOp->getLoc());
    auto ctx = this->getContext();
    const auto hasEltwiseTrait = eltwiseOp->template hasTrait<IE::EltwiseOp>();
    const auto isGroupConv = mlir::isa<IE::GroupConvolutionOp>(eltwiseOp);
    if (!hasEltwiseTrait && !isGroupConv) {
        return mlir::failure();
    }

    if (isGroupConv) {
        mlir::SmallVector<Const::DeclareOp> constInputOps;
        // Skip the first operand, check only weights and biases.
        for (unsigned operandIdx = 1; operandIdx < eltwiseOp->getNumOperands(); operandIdx++) {
            const mlir::Value operand = eltwiseOp->getOperand(operandIdx);
            Const::DeclareOp declareOp = operand.getDefiningOp<Const::DeclareOp>();
            constInputOps.push_back(declareOp);
        }
        const auto supportedGroupConvConst = llvm::all_of(constInputOps, isSplatConstant);
        if (!supportedGroupConvConst) {
            return mlir::failure();
        }
    }

    SmallVector<IE::MemPermuteOp> inputMemPermutes;
    const unsigned numInputs = isGroupConv ? 1 : 2;
    for (unsigned inIdx = 0; inIdx < numInputs; inIdx++) {
        inputMemPermutes.push_back(getEltwiseInputPermute(eltwiseOp->getOperand(inIdx)));
    }
    const auto isMemPermute = [](const IE::MemPermuteOp op) -> bool {
        return op != nullptr;
    };
    auto allInputsArePermutes = std::all_of(inputMemPermutes.begin(), inputMemPermutes.end(), isMemPermute);
    if (!allInputsArePermutes) {
        return mlir::failure();
    }
    SmallVector<vpux::NDTypeInterface> permuteInputTypes;
    const auto getInputType = [](mlir::Operation* permute) -> vpux::NDTypeInterface {
        return permute->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
    };
    std::transform(inputMemPermutes.begin(), inputMemPermutes.end(), std::back_inserter(permuteInputTypes),
                   getInputType);

    SmallVector<DimsOrder> permuteInputLayouts;
    const auto getInputLayout = [](mlir::Operation* permute) -> DimsOrder {
        return DimsOrder::fromValue(permute->getOperand(0));
    };
    std::transform(inputMemPermutes.begin(), inputMemPermutes.end(), std::back_inserter(permuteInputLayouts),
                   getInputLayout);

    SmallVector<DimsOrder> permuteMemPermLayouts;
    const auto getMemPermLayout = [](IE::MemPermuteOp permute) -> DimsOrder {
        return DimsOrder::fromAffineMap(permute.mem_permAttr().getValue());
    };
    std::transform(inputMemPermutes.begin(), inputMemPermutes.end(), std::back_inserter(permuteMemPermLayouts),
                   getMemPermLayout);

    auto eltwiseInput1Type = eltwiseOp->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
    auto eltwiseInputLayout = eltwiseInput1Type.getDimsOrder();
    auto eltwiseInputShape = eltwiseInput1Type.getShape();
    auto eltwiseOutputLayout = eltwiseOp.output().getType().template cast<vpux::NDTypeInterface>().getDimsOrder();
    auto patternCanBeConverted = [&]() -> bool {
        const auto firstInputLayout = permuteInputLayouts[0];
        const auto isSameLayout = [firstInputLayout](const DimsOrder layout) -> bool {
            return firstInputLayout == layout;
        };
        if (!std::all_of(permuteInputLayouts.begin(), permuteInputLayouts.end(), isSameLayout)) {
            // If the two inputs before Permutes have different layouts, the case is not supported
            // e.g., input1 layout is NCHW, input2 layout is NWHC
            return false;
        }
        if (eltwiseInputLayout != DimsOrder::NHWC) {
            // Consider the layout is adjusted to NHWC for Eltwise
            return false;
        }
        if (eltwiseOutputLayout != eltwiseInputLayout) {
            return false;
        }
        const auto firstMemPermLayout = permuteMemPermLayouts[0];
        const auto isSameMemPerm = [firstMemPermLayout](const DimsOrder layout) -> bool {
            return firstMemPermLayout == layout;
        };
        if (!std::all_of(permuteMemPermLayouts.begin(), permuteMemPermLayouts.end(), isSameMemPerm)) {
            return false;
        }

        auto output = eltwiseOp.output();
        // When elementwise operation has two consumers with at least one IE.ShapeCast, skip such case.
        // Such branching leads to compilation failure.
        const auto isShapeCast = [](mlir::Operation* op) -> bool {
            return mlir::isa<IE::ShapeCastOp>(op);
        };
        const auto hasShapeCastConsumer = llvm::any_of(output.getUsers(), isShapeCast);
        if (!output.hasOneUse() && hasShapeCastConsumer) {
            return false;
        }

        return true;
    };
    if (!patternCanBeConverted()) {
        return mlir::failure();
    }
    _log.nest().trace("Moving permute op post eltwise {0} at {1}", eltwiseOp->getName(), eltwiseOp->getLoc());
    auto permutesToMove = getPermutesToMove(inputMemPermutes);
    const auto neutralMemPerm = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap({0, 1, 2, 3}, ctx));
    // Get operation skipping QuantizeCast and ShapeCast ops
    auto getOwnerIgnoreCasts = [&](const mlir::OpOperand& opOperand) -> mlir::Operation* {
        auto ownerOp = opOperand.getOwner();
        while (ownerOp && mlir::isa<IE::QuantizeCastOp, IE::ShapeCastOp>(ownerOp) &&
               !ownerOp->getResult(0).getUsers().empty()) {
            ownerOp = *ownerOp->getResult(0).getUsers().begin();
        }
        return ownerOp;
    };
    auto mappedShape = getMappedShape(permuteInputLayouts[0], eltwiseInputLayout, permuteMemPermLayouts[0],
                                      getShape(inputMemPermutes[0].input()));
    for (auto curPermute : permutesToMove) {
        _log.nest().trace("Processing permute {0} {1}", curPermute->getName(), curPermute->getLoc());
        auto permuteOutputType = curPermute.output().getType().template cast<vpux::NDTypeInterface>();
        const auto dstOrder = mlir::AffineMapAttr::get(eltwiseInputLayout.toAffineMap(ctx));
        rewriter.setInsertionPoint(curPermute);
        if (permuteInputLayouts[0] != eltwiseInputLayout) {
            auto permuteCast = rewriter.template create<IE::PermuteCastOp>(curPermute->getLoc(), curPermute.input(),
                                                                           dstOrder, neutralMemPerm);
            auto newPermuteCastOutputType = permuteCast.output().getType().template cast<vpux::NDTypeInterface>();
            mappedShape = newPermuteCastOutputType.getShape().toValues();
            auto shapeCast = rewriter.template create<IE::ShapeCastOp>(
                    curPermute->getLoc(), newPermuteCastOutputType.changeShape(eltwiseInputShape), permuteCast.output(),
                    getIntArrayAttr(ctx, eltwiseInputShape.raw()));
            curPermute.output().replaceUsesWithIf(shapeCast.result(), [&](mlir::OpOperand& opOperand) {
                return getOwnerIgnoreCasts(opOperand) == eltwiseOp;
            });
        } else {
            auto shapeCast = rewriter.template create<IE::ShapeCastOp>(
                    curPermute->getLoc(), permuteOutputType.changeShape(eltwiseInputShape), curPermute.input(),
                    getIntArrayAttr(ctx, eltwiseInputShape.raw()));
            curPermute.output().replaceUsesWithIf(shapeCast.result(), [&](mlir::OpOperand& opOperand) {
                return getOwnerIgnoreCasts(opOperand) == eltwiseOp;
            });
        }
    }

    // Get the output value of the last QuantizeCast op
    // or the output value of the Eltwise op if no QuantizeCast op exists
    auto getInsertPoint = [&]() -> mlir::Value {
        auto output = eltwiseOp.output();
        if (output.getUsers().empty()) {
            return output;
        }
        // When there are more than one consumer, return the output eltwise.
        if (!output.hasOneUse()) {
            return output;
        }
        if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*output.getUsers().begin())) {
            return quantizeCastOp.output();
        }
        if (auto shapeCastOp = mlir::dyn_cast<IE::ShapeCastOp>(*output.getUsers().begin())) {
            // Same here, when IE.ShapeCast has several consumers, bail out.
            if (!shapeCastOp.result().hasOneUse()) {
                return shapeCastOp.result();
            }
            // In case IE.Add -> IE.ShapeCast -> IE.QuantizeCast, return QuantizeCast.
            if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*shapeCastOp.result().getUsers().begin())) {
                return quantizeCastOp.output();
            }

            return shapeCastOp.result();
        }
        return output;
    };
    auto outputValue = getInsertPoint();
    auto eltwiseOutputType = outputValue.getType().template cast<vpux::NDTypeInterface>();
    rewriter.setInsertionPointAfter(outputValue.getDefiningOp());
    auto newOutputShapeCastType = eltwiseOutputType.changeShape(mappedShape);
    if (permuteInputLayouts[0] != eltwiseInputLayout) {
        auto outputShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType, outputValue, getIntArrayAttr(ctx, mappedShape.raw()));
        const auto outputDstOrder = mlir::AffineMapAttr::get(permuteInputLayouts[0].toAffineMap(ctx));
        auto outputPermuteCast = rewriter.template create<IE::PermuteCastOp>(
                eltwiseOp->getLoc(), outputShapeCast.result(), outputDstOrder, neutralMemPerm);
        auto outputPermute = rewriter.template create<IE::MemPermuteOp>(eltwiseOp->getLoc(), outputPermuteCast.output(),
                                                                        inputMemPermutes[0].dst_orderAttr(),
                                                                        inputMemPermutes[0].mem_permAttr());
        auto eltwiseOutputShape = eltwiseOutputType.getShape();
        auto outputPermuteOutputShape = newOutputShapeCastType.getShape();
        if (eltwiseOutputShape != outputPermuteOutputShape) {
            /*
            For the specific case

               Permute    Permute
                  |          |
              ShapeCast   ShapeCast
                  |          |
            (QuantizeCast) (QuantizeCast)
                    \        /
                     Eltwise
                        |
                     Eltwise
                        |

                to:

            PermuteCast  PermuteCast
                  |          |
             ShapeCast   ShapeCast
                  |          |
            (QuantizeCast) (QuantizeCast)
                   \        /
                     Eltwise
                        |
                 (QuantizeCast)
                        |
                    ShapeCast
                        |
                   PermuteCast
                        |
                     Permute
                        |
                    ShapeCast
             There should be an extra ShapeCast.
            */
            auto outputPermuteShapeCast = rewriter.template create<IE::ShapeCastOp>(
                    eltwiseOp->getLoc(), eltwiseOutputType, outputPermute.output(),
                    getIntArrayAttr(ctx, eltwiseOutputShape.raw()));
            outputValue.replaceAllUsesExcept(outputPermuteShapeCast, outputShapeCast);
        } else {
            outputPermute.output().setType(eltwiseOutputType);
            outputValue.replaceAllUsesExcept(outputPermute, outputShapeCast);
        }
    } else {
        auto outputShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType.changeShape(permuteInputTypes[0].getShape()), outputValue,
                getIntArrayAttr(ctx, permuteInputTypes[0].getShape().raw()));
        auto outputPermute = rewriter.template create<IE::MemPermuteOp>(eltwiseOp->getLoc(), outputShapeCast.result(),
                                                                        inputMemPermutes[0].dst_orderAttr(),
                                                                        inputMemPermutes[0].mem_permAttr());
        outputPermute.output().setType(eltwiseOutputType);
        outputValue.replaceAllUsesExcept(outputPermute, outputShapeCast);
    }

    return mlir::success();
}

void MovePermutePostEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<PermuteEltwiseRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createMovePermutePostEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMovePermutePostEltwisePass(Logger log) {
    return std::make_unique<MovePermutePostEltwisePass>(log);
}

//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <vector>

using namespace vpux;

namespace {

bool isFQAgnostic(mlir::Operation* op) {
    // TODO: #18651 replace with mlir::isa<IE::ElemTypeInfoOpInterface>(op);
    return (mlir::isa<IE::ConcatOp, IE::ReshapeOp, IE::SplitOp, IE::TileOp, IE::MaxPoolOp, IE::ReduceMaxOp, IE::SliceOp,
                      IE::TransposeOp>(op));
}

SmallVector<mlir::Operation*> gatherNodesAround(mlir::Operation* op) {
    auto aroundOps = SmallVector<mlir::Operation*>();

    for (const auto& operand : op->getOperands()) {
        if (operand.getDefiningOp() == nullptr) {
            continue;
        }
        aroundOps.push_back(operand.getDefiningOp());
    }
    for (auto results : op->getResults()) {
        for (auto user : results.getUsers()) {
            aroundOps.push_back(user);
        }
    }
    return aroundOps;
}

void gatherFQs(mlir::Operation* currentOp, SmallVector<IE::FakeQuantizeOp>& fqOpsToAlign,
               SmallVector<mlir::Operation*>& visitedFQAgnosticOps) {
    // Mark current operation as visited
    visitedFQAgnosticOps.push_back(currentOp);
    for (const auto& operand : currentOp->getOperands()) {
        auto fqOpToAlign = operand.getDefiningOp<IE::FakeQuantizeOp>();
        if (fqOpToAlign == nullptr) {
            if (operand.getDefiningOp() == nullptr || !isFQAgnostic(operand.getDefiningOp()) ||
                llvm::find(visitedFQAgnosticOps, operand.getDefiningOp()) != visitedFQAgnosticOps.end()) {
                continue;
            }
            gatherFQs(operand.getDefiningOp(), fqOpsToAlign, visitedFQAgnosticOps);
            continue;
        }

        if (llvm::find(fqOpsToAlign, fqOpToAlign) != fqOpsToAlign.end()) {
            continue;
        }

        fqOpsToAlign.push_back(fqOpToAlign);

        auto opsAround = gatherNodesAround(fqOpToAlign.getOperation());
        for (auto opAround : opsAround) {
            if (!isFQAgnostic(opAround) || llvm::find(visitedFQAgnosticOps, opAround) != visitedFQAgnosticOps.end()) {
                continue;
            }
            gatherFQs(opAround, fqOpsToAlign, visitedFQAgnosticOps);
        }
    }

    if (!isFQAgnostic(currentOp)) {
        return;
    }

    for (const auto& results : currentOp->getResults()) {
        for (auto user : results.getUsers()) {
            auto fqOpToAlign = mlir::dyn_cast<IE::FakeQuantizeOp>(user);
            if (fqOpToAlign == nullptr) {
                if (!isFQAgnostic(user) || llvm::find(visitedFQAgnosticOps, user) != visitedFQAgnosticOps.end()) {
                    continue;
                }
                gatherFQs(user, fqOpsToAlign, visitedFQAgnosticOps);
                continue;
            }

            if (llvm::find(fqOpsToAlign, fqOpToAlign) != fqOpsToAlign.end()) {
                continue;
            }

            fqOpsToAlign.push_back(fqOpToAlign);
            auto opsAround = gatherNodesAround(user);
            for (auto opAround : opsAround) {
                if (!isFQAgnostic(opAround) ||
                    llvm::find(visitedFQAgnosticOps, opAround) != visitedFQAgnosticOps.end()) {
                    continue;
                }
                gatherFQs(opAround, fqOpsToAlign, visitedFQAgnosticOps);
            }
        }
    }
}

bool isPerTensorFQ(ArrayRef<IE::FakeQuantizeOp> fqOpsToAlign) {
    for (const auto& fqOpToAlign : fqOpsToAlign) {
        const auto axis = IE::getFQAxisIndex(fqOpToAlign);
        if (axis != None && axis.hasValue()) {
            return false;
        }
    }
    return true;
}

Const::details::ContentRange<float> getConst(Const::DeclareOp declOp) {
    const auto content = declOp.contentAttr().fold();
    return content.getValues<float>();
}

bool allFqsHaveSameIOParams(MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign) {
    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto inputLowVals = getConst(fqOpToAlign.input_low().getDefiningOp<Const::DeclareOp>());
        const auto inputHighVals = getConst(fqOpToAlign.input_high().getDefiningOp<Const::DeclareOp>());
        const auto outLowVals = getConst(fqOpToAlign.output_low().getDefiningOp<Const::DeclareOp>());
        const auto outHighVals = getConst(fqOpToAlign.output_high().getDefiningOp<Const::DeclareOp>());

        auto fqInValsCount = inputLowVals.size();

        if (fqInValsCount != inputHighVals.size()) {
            return false;
        }

        auto fqOutValsCount = outLowVals.size();
        if (fqOutValsCount != outHighVals.size()) {
            return false;
        }

        if (fqInValsCount != fqOutValsCount) {
            return false;
        }

        for (size_t i = 0; i < fqInValsCount; i++) {
            if (!isFloatEqual(inputLowVals[i], outLowVals[i]) || !isFloatEqual(inputHighVals[i], outHighVals[i])) {
                return false;
            }
        }
    }
    return true;
}

bool allFqsHaveTheSameRange(MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign) {
    const auto firstFQInLowVals = getConst(fqOpsToAlign[0].input_low().getDefiningOp<Const::DeclareOp>());
    const auto firstFQInHighVals = getConst(fqOpsToAlign[0].input_high().getDefiningOp<Const::DeclareOp>());

    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto inFQLowVals = getConst(fqOpToAlign.input_low().getDefiningOp<Const::DeclareOp>());
        const auto inFQHighVals = getConst(fqOpToAlign.input_high().getDefiningOp<Const::DeclareOp>());

        if (!isFloatEqual(firstFQInLowVals[0], inFQLowVals[0]) ||
            !isFloatEqual(firstFQInHighVals[0], inFQHighVals[0])) {
            return false;
        }
    }
    return true;
}

int64_t calculateZeroPoint(float low, float high, int levels, mlir::IntegerType type) {
    VPUX_THROW_UNLESS((low <= 0.f) && (high >= 0.f) && (low != high), "Wrong low and high values.");
    VPUX_THROW_UNLESS(levels <= 256, "Levels must be less then 256.");

    int64_t zeroPoint = 0;

    if (type.isUnsignedInteger()) {
        float x = -static_cast<float>(levels - 1) * low / (high - low);
        zeroPoint = static_cast<int>(std::round(x));
    } else if (type.isSignedInteger()) {
        float x = -static_cast<float>(levels - 1) * ((high + low) * 0.5f) / (high - low);
        zeroPoint = static_cast<int>(std::round(x));
    } else {
        VPUX_THROW("Unsupported element type {0}.", type);
    }

    return zeroPoint;
}

void alignZP(float& min, float& max, int maxLevels, mlir::IntegerType type) {
    auto zp = calculateZeroPoint(min, max, maxLevels, type);
    auto scale = (max - min) / (maxLevels - 1);
    min = scale * (-zp);
    max = static_cast<float>(scale * (maxLevels - 1.0 - zp));
}

Const::DeclareOp createFQConst(mlir::MLIRContext* ctx, mlir::Location loc, float val, mlir::RankedTensorType argType,
                               mlir::PatternRewriter& rewriter) {
    const auto denseElementVal = wrapData(mlir::RankedTensorType::get({1, 1, 1, 1}, mlir::Float32Type::get(ctx)), val);
    VPUX_THROW_UNLESS(denseElementVal != nullptr, "Failed to generate the denseElementVal.");
    Const::ContentAttr cstAttr = Const::ContentAttr::get(denseElementVal)
                                         .convertElemType(argType.cast<vpux::NDTypeInterface>().getElementType());
    return rewriter.create<Const::DeclareOp>(loc, argType, cstAttr);
}

void adjustFqsToAlign(SmallVector<IE::FakeQuantizeOp>& fqOpsToAlign) {
    auto minRange = std::numeric_limits<float>::max();
    SmallVector<IE::FakeQuantizeOp> filteredFQOpsToAlign;
    const float maxFqRangeRatio = 5.0;

    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto inputLowVals = getConst(fqOpToAlign.input_low().getDefiningOp<Const::DeclareOp>());
        const auto inputHighVals = getConst(fqOpToAlign.input_high().getDefiningOp<Const::DeclareOp>());

        minRange = std::min(minRange, inputHighVals[0] - inputLowVals[0]);
    }

    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto inputLowVals = getConst(fqOpToAlign.input_low().getDefiningOp<Const::DeclareOp>());
        const auto inputHighVals = getConst(fqOpToAlign.input_high().getDefiningOp<Const::DeclareOp>());

        if (inputHighVals[0] - inputLowVals[0] >= maxFqRangeRatio * minRange) {
            continue;
        }
        filteredFQOpsToAlign.push_back(fqOpToAlign);
    }
    fqOpsToAlign = filteredFQOpsToAlign;
}

void findMinMax(MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign, float& min, float& max, float& range,
                int& maxLevels) {
    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto inputLowVals = getConst(fqOpToAlign.input_low().getDefiningOp<Const::DeclareOp>());
        const auto inputHighVals = getConst(fqOpToAlign.input_high().getDefiningOp<Const::DeclareOp>());

        maxLevels = std::max(maxLevels, static_cast<int>(fqOpToAlign.levels()));

        for (size_t i = 0; i < inputLowVals.size(); i++) {
            min = std::min(min, inputLowVals[i]);
            max = std::max(max, inputHighVals[i]);
            range = std::min(range, inputHighVals[i] - inputLowVals[i]);
        }
    }
}

void alignFQRanges(IE::ConcatOp origOp, MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign,
                   mlir::PatternRewriter& rewriter, const float min, const float max, const int maxLevels, Logger log) {
    log.trace("alignFQRanges for {0}", origOp->getLoc());
    mlir::MLIRContext* ctx = origOp->getContext();

    // Determine the insertion point for the common Const::DeclareOps that containg the align FQ ranges
    auto theUppestFQOp = fqOpsToAlign[0].getOperation();
    for (size_t i = 1; i < fqOpsToAlign.size(); i++) {
        if (!theUppestFQOp->isBeforeInBlock(fqOpsToAlign[i])) {
            theUppestFQOp = fqOpsToAlign[i];
        }
    }
    log.trace("Set insertion point for common constants at {0}", theUppestFQOp->getLoc());
    rewriter.setInsertionPoint(theUppestFQOp);
    // Create the input/output low/high constants and level attribute that will be used to align the FQ ranges
    const auto elemType = fqOpsToAlign[0].input().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto fqArgType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);
    auto commonInputLow = createFQConst(ctx, origOp->getLoc(), min, fqArgType, rewriter);
    auto commonInputHigh = createFQConst(ctx, origOp->getLoc(), max, fqArgType, rewriter);
    auto commonLevels = getIntAttr(rewriter, maxLevels);
    log.trace("Created common constants for input/output low/high constants and level attribute.");

    // Search for the constants containing the min and max of the FQ range from the list of FQs in order to not create
    // them again
    for (size_t i = 0; i < fqOpsToAlign.size(); i++) {
        const auto inputLowVals = getConst(fqOpsToAlign[i].input_low().getDefiningOp<Const::DeclareOp>());
        const auto inputHighVals = getConst(fqOpsToAlign[i].input_high().getDefiningOp<Const::DeclareOp>());
        auto fqHasMaxRanges = [min, max, fqOpsToAlign, maxLevels, inputLowVals,
                               inputHighVals](IE::FakeQuantizeOp fqOp) -> bool {
            return (static_cast<int>(fqOp.levels()) == maxLevels && isFloatEqual(min, inputLowVals[0]) &&
                    isFloatEqual(max, inputHighVals[0]));
        };

        if (fqHasMaxRanges(fqOpsToAlign[i])) {
            continue;
        }

        log.trace("Align ranges of FQ at {0}", fqOpsToAlign[i]->getLoc());

        const auto inputLowAttr = getFPAttr(ctx, inputLowVals[0]);
        const auto inputHighAttr = getFPAttr(ctx, inputHighVals[0]);

        // Set insertion point at the FQ that will be rewritten
        rewriter.setInsertionPoint(fqOpsToAlign[i]);
        // Replace the old FQ operation with the FQ with new ranges + Clamp operation that preserves the original
        // range interval
        auto newFQOp = rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), fqOpsToAlign[i].input(), commonInputLow,
                                                           commonInputHigh, commonInputLow, commonInputHigh,
                                                           commonLevels, fqOpsToAlign[i].auto_broadcastAttr());
        log.trace("Created new FQ op at {0}", newFQOp->getLoc());

        // Replace old FQ with new Clamp that preserve the old FQ's quantization ranges
        auto clampOp = rewriter.replaceOpWithNewOp<IE::ClampOp>(fqOpsToAlign[i], newFQOp.output(), inputLowAttr,
                                                                inputHighAttr);
        log.trace("Created new Clamp op at {0}", clampOp->getLoc());
    }
}

class AlignConcatScalesRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    AlignConcatScalesRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx, benefitLow), _log(log) {
        setDebugName("AlignConcatScalesRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AlignConcatScalesRewriter::matchAndRewrite(IE::ConcatOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = origOp->getContext();

    // Align FQ ranges and insert Clamp to restore ranges from FQ input
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());

    SmallVector<IE::FakeQuantizeOp> fqOpsToAlign;
    SmallVector<mlir::Operation*> visitedFQAgnosticOps;

    gatherFQs(origOp, fqOpsToAlign, visitedFQAgnosticOps);
    _log.trace("Gather {0} FQs to align.", fqOpsToAlign.size());

    // 1. Check if is per tensor FQ
    if (!isPerTensorFQ(fqOpsToAlign)) {
        _log.trace("Failed to align FQ ranges due to per channel FQ");
        return mlir::failure();
    }

    // 2. Check that there are at least 2 FQs to align
    if (fqOpsToAlign.size() < 2) {
        _log.trace("Failed because there are only {0} FQs to align. At least two needed.", fqOpsToAlign.size());
        return mlir::failure();
    }

    // 3. Check that input and output ranges are equal for each FQ
    if (!allFqsHaveSameIOParams(fqOpsToAlign)) {
        _log.trace("Failed because the FQs have different ranges between input and output.");
        return mlir::failure();
    }

    // 4. Keep only the FQs whose ranges are relatively close in terms of intervals
    adjustFqsToAlign(fqOpsToAlign);
    if (fqOpsToAlign.size() < 2) {
        _log.trace("Failed because the FQs have too different quantization ranges.");
        return mlir::failure();
    }

    // 5. Check that the ranges are already aligned
    if (allFqsHaveTheSameRange(fqOpsToAlign)) {
        _log.trace("All gathered FQs have the same FQs.");
        return mlir::failure();
    }

    _log.trace("All FQ alignment conditions are met.");

    float min = 0;
    float max = 0;
    float range = 0;
    int maxLevels = 0;

    // Determine the alignment interval limits
    findMinMax(fqOpsToAlign, min, max, range, maxLevels);
    alignZP(min, max, maxLevels, mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned));
    _log.trace("Min = {0}, Max = {1}, MaxLevels = {2}.", min, max, maxLevels);
    alignFQRanges(origOp, fqOpsToAlign, rewriter, min, max, maxLevels, _log);
    return mlir::success();
}

class AlignScalesPass final : public IE::AlignScalesBase<AlignScalesPass> {
public:
    explicit AlignScalesPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void AlignScalesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AlignConcatScalesRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAlignScalesPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createAlignScalesPass(Logger log) {
    return std::make_unique<AlignScalesPass>(log);
}

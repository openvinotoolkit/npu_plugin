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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <legacy/ngraph_ops/convolution_ie.hpp>

using namespace vpux;

namespace {

//
// ReplacePadWithConstAndConcat
//

class ReplacePadWithConstAndConcat final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    ReplacePadWithConstAndConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
        setDebugName("ReplacePadWithConstAndConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp origPadOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::ArrayAttr padValue) {
    if (padValue == nullptr) {
        return mlir::failure();
    }

    const auto valueVector = parseIntArrayAttr<int64_t>(padValue);

    if (valueVector.size() != 4) {
        return mlir::failure();
    }

    return valueVector;
}

static mlir::DenseElementsAttr buildWeightData(const mlir::RankedTensorType dataStorageType, const float value) {
    const auto elemType = dataStorageType.getElementType();
    if (elemType.isF32()) {
        return mlir::DenseElementsAttr::get(dataStorageType, value);
    } else if (elemType.isF16()) {
        const ngraph::float16 valueFP16 = value;
        return mlir::DenseElementsAttr::get(dataStorageType, valueFP16);
    } else if (elemType.isInteger(CHAR_BIT)) {
        const uint8_t valueU8 = checked_cast<uint8_t>(value);
        return mlir::DenseElementsAttr::get(dataStorageType, valueU8);
    }
    return nullptr;
}

mlir::LogicalResult ReplacePadWithConstAndConcat::matchAndRewrite(IE::PadOp origPadOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::PadOp Operation '{0}'", origPadOp->getLoc());

    if (origPadOp.mode() != IE::PadMode::CONSTANT) {
        return mlir::failure();
    }

    auto padsBegin = extractPads(origPadOp.pads_begin_attrAttr());
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }

    auto padsEnd = extractPads(origPadOp.pads_end_attrAttr());
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(origPadOp.pad_value_attr().hasValue(), "IE::PadOp has pad_value_attr() == nullptr {0}",
                      origPadOp->getLoc());
    auto padsValue = origPadOp.pad_value_attr().getValue().convertToFloat();

    const auto inputShape = origPadOp.input().getType().cast<mlir::ShapedType>().getShape();
    const auto outputShape = origPadOp.output().getType().cast<mlir::ShapedType>().getShape();

    const auto createConstOp = [&](SmallVector<mlir::Value>& values, const size_t ind,
                                   const SmallVector<int64_t>& padSize, const float padValue) {
        if (padSize[ind] != 0) {
            auto constShape = padSize;
            for (const auto seg : irange(inputShape.size())) {
                constShape[seg] = seg < ind ? inputShape[seg] : outputShape[seg];
            }
            constShape[ind] = padSize[ind];

            const auto elemType = origPadOp.input().getType().cast<mlir::ShapedType>().getElementType();
            const auto dataStorageType = mlir::RankedTensorType::get(constShape, elemType);
            const auto dataAttr = buildWeightData(dataStorageType, padValue);
            VPUX_THROW_UNLESS(
                    dataAttr != nullptr,
                    "`IE::PadOp` has incompatible data type {0}. Only uint8 or float16 or float32 are supported",
                    elemType);

            auto constant = rewriter.create<Const::DeclareOp>(origPadOp->getLoc(), dataStorageType,
                                                              Const::ContentAttr::get(dataAttr));

            values.push_back(constant);
        }
    };

    mlir::Value midInput = origPadOp.input();
    for (const auto ind : irange(inputShape.size())) {
        if (padsBegin.getValue()[3 - ind] == 0 && padsEnd.getValue()[3 - ind] == 0) {
            continue;
        }

        SmallVector<mlir::Value> valueRange;

        _log.nest().trace("Insert ConstOp convert from padsBegin index: {0}", (3 - ind));
        createConstOp(valueRange, (3 - ind), padsBegin.getValue(), padsValue);

        valueRange.push_back(midInput);

        _log.nest().trace("Insert ConstOp convert from padsEnd index: {0}", (3 - ind));
        createConstOp(valueRange, (3 - ind), padsEnd.getValue(), padsValue);

        auto concat = rewriter.create<IE::ConcatOp>(midInput.getLoc(), valueRange, (3 - ind));
        _log.nest().trace("Insert ConcatOp {0}", concat.getLoc());
        midInput = concat.output();
    }
    const auto userOp = *origPadOp.output().getUsers().begin();
    userOp->setOperand(0, midInput);
    rewriter.eraseOp(origPadOp);

    return mlir::success();
}

//
// ConvertPadToConcat
//

class ConvertPadToConcatPass final : public IE::ConvertPadToConcatBase<ConvertPadToConcatPass> {
public:
    explicit ConvertPadToConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertPadToConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ReplacePadWithConstAndConcat>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSupportFusePadOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPadToConcatPass(Logger log) {
    return std::make_unique<ConvertPadToConcatPass>(log);
}

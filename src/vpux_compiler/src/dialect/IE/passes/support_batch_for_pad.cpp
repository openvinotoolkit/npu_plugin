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

#include <ngraph_ops/convolution_ie.hpp>

using namespace vpux;

namespace {

//
// AppendPadWithConstants
//

class AppendPadWithConstants final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    AppendPadWithConstants(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
        setDebugName("AppendPadWithConstants");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

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

struct PadWithBatch {
    mlir::ArrayAttr attr;
    int64_t batch;
};

PadWithBatch extractBatchFromPad(mlir::MLIRContext* ctx, mlir::ArrayAttr pad) {
    auto values = parseIntArrayAttr<int64_t>(pad);
    VPUX_THROW_UNLESS(values.size() == 4, "To extract batch dimension, attribute size should be equal to 4");
    const auto batch = std::exchange(values[0], 0);
    const auto newArrayAttr = getIntArrayAttr(ctx, values);
    return {newArrayAttr, batch};
}

mlir::LogicalResult AppendPadWithConstants::matchAndRewrite(IE::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::PadOp Operation '{0}'", origOp->getLoc());

    if (origOp.mode() != IE::PadMode::CONSTANT) {
        return matchFailed(rewriter, origOp, "IE::PadOp {0} should have Constant mode", origOp);
    }

    VPUX_THROW_UNLESS(origOp.pads_begin_attr().hasValue(), "IE::PadOp has pads_begin_attr() == nullptr {0}",
                      origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.pads_end_attr().hasValue(), "IE::PadOp has pads_end_attr() == nullptr {0}",
                      origOp->getLoc());

    const auto newPadsBegin = extractBatchFromPad(getContext(), origOp.pads_begin_attr().getValue());
    const auto newPadsEnd = extractBatchFromPad(getContext(), origOp.pads_end_attr().getValue());
    const auto outShape = origOp.output().getType().cast<mlir::ShapedType>().getShape();

    auto batchlessPadOp = rewriter.create<IE::PadOp>(origOp->getLoc(), origOp.input(), origOp.pads_begin(),
                                                     origOp.pads_end(), origOp.pad_value(), newPadsBegin.attr,
                                                     newPadsEnd.attr, origOp.pad_value_attrAttr(), origOp.modeAttr());

    const auto createConstOp = [&](SmallVector<mlir::Value>& values, const PadWithBatch& newPad) {
        if (auto batch = newPad.batch) {
            const auto value = checked_cast<float>(origOp.pad_value_attrAttr().getValueAsDouble());
            const auto elemType = origOp.input().getType().cast<mlir::ShapedType>().getElementType();
            const auto constShape = SmallVector<int64_t>{batch, outShape[1], outShape[2], outShape[3]};
            const auto dataStorageType = mlir::RankedTensorType::get(constShape, elemType);
            const auto dataAttr = buildWeightData(dataStorageType, value);
            VPUX_THROW_UNLESS(
                    dataAttr != nullptr,
                    "`IE::PadOp` has incompatible data type {0}. Only uint8 or float16 or float32 are supported",
                    elemType);

            auto constant = rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataStorageType,
                                                              Const::ContentAttr::get(dataAttr));
            values.push_back(constant);
        }
    };

    SmallVector<mlir::Value> valueRange;

    _log.nest().trace("Insert {0} batches before Pad", newPadsBegin.batch);
    createConstOp(valueRange, newPadsBegin);

    _log.nest().trace("Insert batchless Pad {0}", batchlessPadOp.getLoc());
    valueRange.push_back(batchlessPadOp.output());

    _log.nest().trace("Insert {0} batches after Pad", newPadsBegin.batch);
    createConstOp(valueRange, newPadsEnd);

    auto concat = rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, valueRange, 0);
    _log.nest().trace("Replaced with ConcatOp {0}", concat.getLoc());

    return mlir::success();
}

//
// SupportBatchForPadPass
//

class SupportBatchForPadPass final : public IE::SupportBatchForPadBase<SupportBatchForPadPass> {
public:
    explicit SupportBatchForPadPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SupportBatchForPadPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalPadOp = [&](IE::PadOp pad) {
        const auto beginsShape = parseIntArrayAttr<int64_t>(pad.pads_begin_attr().getValue());
        const auto endsShape = parseIntArrayAttr<int64_t>(pad.pads_end_attr().getValue());
        const auto pads4D = (beginsShape.size() == 4) && (endsShape.size() == 4);
        if (!pads4D) {
            return true;
        }
        const auto zeroBatchPads = (beginsShape[0] == 0 && endsShape[0] == 0);
        return zeroBatchPads;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalOp<IE::PadOp>(isLegalPadOp);
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<AppendPadWithConstants>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSupportBatchForPadPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSupportBatchForPadPass(Logger log) {
    return std::make_unique<SupportBatchForPadPass>(log);
}

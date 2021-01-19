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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/FakeQuantSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvertIE2QuantPass
//

class ConvertIE2QuantPass final : public ConvertIE2QuantBase<ConvertIE2QuantPass> {
public:
    explicit ConvertIE2QuantPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class UseConstFakeQuant;
    class UseDequantize;

public:
    static const mlir::PatternBenefit lowBenefit;
    static const mlir::PatternBenefit highBenefit;

public:
    struct ConstFakeQuantParams final {
        uint32_t num_bits = 0;
        bool narrow_range = false;
        // TODO: is it possible to get this value from IE FakeQuantize?
        bool is_signed = false;

        double minVal = 0.0;
        double maxVal = 0.0;

        Optional<int32_t> axisInd;
        SmallVector<double> minVals;
        SmallVector<double> maxVals;
    };

public:
    static mlir::FailureOr<ConstFakeQuantParams> getConstFakeQuantParams(IE::FakeQuantizeOp origOp, Logger log);

private:
    void passBody();

private:
    Logger _log;
};

const mlir::PatternBenefit ConvertIE2QuantPass::lowBenefit(1);
const mlir::PatternBenefit ConvertIE2QuantPass::highBenefit(2);

void ConvertIE2QuantPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// getConstFakeQuantParams
//

mlir::FailureOr<ConvertIE2QuantPass::ConstFakeQuantParams> ConvertIE2QuantPass::getConstFakeQuantParams(
        IE::FakeQuantizeOp origOp, Logger log) {
    ConstFakeQuantParams params;

    auto lowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto highConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (lowConst == nullptr || highConst == nullptr) {
        log.trace("[{0}] Got non constant parameters \n", origOp->getLoc());
        return mlir::failure();
    }

    switch (origOp.levels()) {
    case 256:
        params.num_bits = 8;
        params.narrow_range = false;
        break;

    case 255:
        params.num_bits = 8;
        params.narrow_range = true;
        break;

    case 16:
        params.num_bits = 4;
        params.narrow_range = false;
        break;

    case 15:
        params.num_bits = 4;
        params.narrow_range = true;
        break;

    default:
        // TODO: support other types?
        log.trace("[{0}] Got unsupported levels '{1}' \n", origOp->getLoc(), origOp.levels());
        return mlir::failure();
    }

    const auto lowAttr = lowConst.getContent();
    const auto highAttr = highConst.getContent();

    if (lowAttr.isSplat() && highAttr.isSplat()) {
        params.minVal = lowAttr.getSplatValue<double>();
        params.maxVal = highAttr.getSplatValue<double>();

        return params;
    } else {
        const auto lowShape = lowConst.getActualType().getShape();
        const auto highShape = highConst.getActualType().getShape();

        if (lowShape != highShape) {
            log.trace("[{0}] Min values shape '{1}' doesn't match with max values shape '{2}' \n", origOp->getLoc(),
                      lowShape, highShape);
            return mlir::failure();
        }

        for (size_t i = 0; i < lowShape.size(); ++i) {
            if (lowShape[i] == 1) {
                continue;
            }

            if (params.axisInd.hasValue()) {
                return mlir::failure();
            }

            params.axisInd = checked_cast<int32_t>(i);
        }
        if (!params.axisInd.hasValue()) {
            log.trace("[{0}] Can't get axis index from shape '{1}' \n", origOp->getLoc(), lowShape);
            return mlir::failure();
        }

        params.minVals = to_small_vector(lowAttr.getValues<double>());
        params.maxVals = to_small_vector(highAttr.getValues<double>());

        return params;
    }
}

//
// UseConstFakeQuant
//

class ConvertIE2QuantPass::UseConstFakeQuant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseConstFakeQuant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx, highBenefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2QuantPass::UseConstFakeQuant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        _log.trace("[{0}] Got non constant parameters \n", origOp->getLoc());
        return mlir::failure();
    }

    const auto outLowAttr = outLowConst.getContent();
    const auto outHighAttr = outHighConst.getContent();

    if (inLowConst.getContent() != outLowAttr || inHighConst.getContent() != outHighAttr) {
        _log.trace("[{0}] Input/output parameters mismatch \n", origOp->getLoc());
        return mlir::failure();
    }

    const auto params = getConstFakeQuantParams(origOp, _log);
    if (mlir::failed(params)) {
        return mlir::failure();
    }

    if (!params->axisInd.hasValue()) {
        rewriter.replaceOpWithNewOp<mlir::quant::ConstFakeQuant>(
                origOp, origOp.output().getType(), origOp.input(),
                getFP32Attr(origOp.getContext(), checked_cast<float>(params->minVal)),
                getFP32Attr(origOp.getContext(), checked_cast<float>(params->maxVal)),
                getInt64Attr(origOp.getContext(), params->num_bits),
                mlir::BoolAttr::get(params->narrow_range, origOp.getContext()),
                mlir::BoolAttr::get(params->is_signed, origOp.getContext()));

        return mlir::success();
    } else {
        const auto minVals = getFP32ArrayAttr(origOp.getContext(), params->minVals);
        const auto maxVals = getFP32ArrayAttr(origOp.getContext(), params->maxVals);

        rewriter.replaceOpWithNewOp<mlir::quant::ConstFakeQuantPerAxis>(
                origOp, origOp.output().getType(), origOp.input(), minVals, maxVals,
                getInt64Attr(origOp.getContext(), params->axisInd.getValue()),
                getInt64Attr(origOp.getContext(), params->num_bits),
                mlir::BoolAttr::get(params->narrow_range, origOp.getContext()),
                mlir::BoolAttr::get(params->is_signed, origOp.getContext()));

        return mlir::success();
    }
}

//
// UseDequantize
//

class ConvertIE2QuantPass::UseDequantize final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx, lowBenefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2QuantPass::UseDequantize::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Got FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inConst = origOp.input().getDefiningOp<ConstantInterface>();
    if (inConst == nullptr) {
        _log.trace("[{0}] Got non constant input \n", origOp->getLoc());
        return mlir::failure();
    }

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        _log.trace("[{0}] Got non constant parameters \n", origOp->getLoc());
        return mlir::failure();
    }

    const auto inLowAttr = inLowConst.getContent();
    const auto inHighAttr = inHighConst.getContent();

    if (!inLowAttr.isSplat() || !inHighAttr.isSplat()) {
        _log.trace("[{0}] Input min/max are not splat values \n", origOp->getLoc());
        return mlir::failure();
    }
    if (inLowAttr.getSplatValue<double>() != 0.0) {
        _log.trace("[{0}] Unsupported input min value '{1}' \n", origOp->getLoc(), inLowAttr.getSplatValue<double>());
        return mlir::failure();
    }
    if (inHighAttr.getSplatValue<double>() > 255.0) {
        _log.trace("[{0}] Unsupported input max value '{1}' \n", origOp->getLoc(), inHighAttr.getSplatValue<double>());
        return mlir::failure();
    }

    const auto params = getConstFakeQuantParams(origOp, _log);
    if (mlir::failed(params)) {
        return mlir::failure();
    }

    const auto actualElemType = inConst.getActualType().getElementType();

    mlir::quant::QuantizedType qType;
    if (!params->axisInd.hasValue()) {
        qType = mlir::quant::fakeQuantAttrsToType(origOp->getLoc(), params->num_bits, params->minVal, params->maxVal,
                                                  params->narrow_range, actualElemType, params->is_signed);
    } else {
        qType = mlir::quant::fakeQuantAttrsToType(origOp->getLoc(), params->num_bits, params->axisInd.getValue(),
                                                  params->minVals, params->maxVals, params->narrow_range,
                                                  actualElemType, params->is_signed);
    }
    if (qType == nullptr) {
        _log.trace("[{0}] Failed to get QuantizedType \n", origOp->getLoc());
        return mlir::failure();
    }

    const auto newInType =
            mlir::RankedTensorType::getChecked(origOp->getLoc(), inConst.getActualType().getShape(), qType);
    if (newInType == nullptr) {
        _log.trace("[{0}] Failed to create Quantized TensorType \n", origOp->getLoc());
        return mlir::failure();
    }

    const auto contentElemType = inConst.getContentType().getElementType();

    IE::ConstantOp newInOp;
    if (contentElemType.isUnsignedInteger(8)) {
        newInOp = rewriter.create<IE::ConstantOp>(inConst->getLoc(), newInType, inConst.getContent());
    } else {
        const auto valsU8 = to_small_vector(inConst.getContent().getValues<uint8_t>());

        const auto newContentType = mlir::RankedTensorType::getChecked(
                origOp->getLoc(), inConst.getContentType().getShape(), getUInt8Type(origOp.getContext()));

        const auto newContent = mlir::DenseElementsAttr::get(newContentType, makeArrayRef(valsU8));

        newInOp = rewriter.create<IE::ConstantOp>(inConst->getLoc(), newInType, newContent);
    }

    rewriter.replaceOpWithNewOp<mlir::quant::DequantizeCastOp>(origOp, origOp.getType(), newInOp.output());

    return mlir::success();
}

//
// passBody
//

void ConvertIE2QuantPass::passBody() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<UseConstFakeQuant>(&ctx, _log.nest());
    patterns.insert<UseDequantize>(&ctx, _log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIE2QuantPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIE2QuantPass(Logger log) {
    return std::make_unique<ConvertIE2QuantPass>(log);
}

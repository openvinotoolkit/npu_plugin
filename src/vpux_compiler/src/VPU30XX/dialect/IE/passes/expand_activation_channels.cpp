//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes/expand_activation_channels.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ExpandActivationChannelsPass
//

class ExpandActivationChannelsPass final :
        public IE::arch30xx::ExpandActivationChannelsBase<ExpandActivationChannelsPass> {
public:
    explicit ExpandActivationChannelsPass(const bool seOpsEnabled, const bool seTransposedConvEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seTransposedConvEnabled(seTransposedConvEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seTransposedConvEnabled;
};

mlir::LogicalResult ExpandActivationChannelsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }
    if (seTransposedConvEnabled.hasValue()) {
        _seTransposedConvEnabled = seTransposedConvEnabled.getValue();
    }

    return mlir::success();
}

void ExpandActivationChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegal = [&](mlir::Operation* op) {
        // TODO E#100360: remove custom flag for IE::TransposedConvolutionOp
        if (!_seTransposedConvEnabled && mlir::isa<IE::TransposedConvolutionOp>(op)) {
            return true;
        }
        if (!_seOpsEnabled && mlir::isa<IE::SEOpInterface>(op) && !mlir::isa<IE::TransposedConvolutionOp>(op)) {
            return true;
        }
        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            return iface.verifyChannels().succeeded();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(isLegal);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::PadOp, IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::MaxPoolRewriter>(&ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::SubtractOp>>(&ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::AndOp>>(&ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<IE::ConvolutionRewriter>(&ctx, _log);
    patterns.add<IE::GroupConvolutionRewriter>(&ctx, _log);

    if (_seOpsEnabled) {
        patterns.add<IE::InterpolateRewriter>(&ctx, _log);
    }
    if (_seTransposedConvEnabled) {
        patterns.add<IE::TransposedConvolutionRewriter>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandActivationChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch30xx::createExpandActivationChannelsPass(const bool seOpsEnabled,
                                                                                   const bool seTransposedConvEnabled,
                                                                                   Logger log) {
    return std::make_unique<ExpandActivationChannelsPass>(seOpsEnabled, seTransposedConvEnabled, log);
}

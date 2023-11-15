//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/VPU30XX/conversion.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public arch30xx::ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvToNCE>(&ctx, arch, _log);
    patterns.add<DepthConvToNCE>(&ctx, arch, _log);
    patterns.add<MaxPoolToNCE>(&ctx, arch, _log);
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, arch, _log);
    patterns.add<EltwiseToNCE<IE::MultiplyOp>>(&ctx, VPU::EltwiseType::MULTIPLY, arch, _log);
    patterns.add<EltwiseToNCE<IE::SubtractOp>>(&ctx, VPU::EltwiseType::SUBTRACT, arch, _log);
    patterns.add<EltwiseToNCE<IE::AndOp>>(&ctx, VPU::EltwiseType::AND, arch, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::arch30xx::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}

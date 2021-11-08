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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_layers_to_EMU.hpp.inc>

//
// ConvertLayers2EMUPass
//

class ConvertLayers2EMUPass final : public ConvertLayers2EMUBase<ConvertLayers2EMUPass> {
public:
    explicit ConvertLayers2EMUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2EMUPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<EMU::EMUDialect>();

    mlir::RewritePatternSet patterns(&ctx);
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2EMUPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2EMUPass(Logger log) {
    return std::make_unique<ConvertLayers2EMUPass>(log);
}

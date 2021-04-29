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

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// Constant locale index manipulation
//

constexpr StringLiteral constLocaleIndex = "const_locale_index";

void setConstLocaleIndex(IERT::ConstantOp op, uint32_t ind) {
    op->setAttr(constLocaleIndex, getInt32Attr(op->getContext(), ind));
}

uint32_t getConstLocaleIndex(mlir::Value val) {
    auto* op = val.getDefiningOp();
    VPUX_THROW_UNLESS(op != nullptr, "Can't get producer for Value '{0}'", val);

    const auto attr = op->getAttrOfType<mlir::IntegerAttr>(constLocaleIndex);
    VPUX_THROW_UNLESS(attr != nullptr, "Can't get attribute '{0}' for Value '{1}'", constLocaleIndex, val);

    return checked_cast<uint32_t>(attr.getInt());
}

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_declarations_to_VPUIP.hpp.inc>

//
// ConvertDeclarations2VPUIPPass
//

class ConvertDeclarations2VPUIPPass final : public ConvertDeclarations2VPUIPBase<ConvertDeclarations2VPUIPPass> {
public:
    explicit ConvertDeclarations2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDeclarations2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    for (auto p : func.getOps<IERT::ConstantOp>() | indexed) {
        setConstLocaleIndex(p.value(), checked_cast<uint32_t>(p.index()));
    }

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::ConcatViewOp, mlir::memref::SubViewOp>();

    mlir::RewritePatternSet patterns(&ctx);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDeclarations2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertDeclarations2VPUIPPass(Logger log) {
    return std::make_unique<ConvertDeclarations2VPUIPPass>(log);
}

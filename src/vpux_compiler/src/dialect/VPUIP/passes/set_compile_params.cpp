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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/generated/schema/gf_version.h"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <version.hpp>

using namespace vpux;

namespace {

//
// SetCompileParamsPass
//

class SetCompileParamsPass final : public VPUIP::SetCompileParamsBase<SetCompileParamsPass> {
public:
    SetCompileParamsPass() = default;
    SetCompileParamsPass(VPUIP::ArchKind arch, VPUIP::CompilationMode compilationMode, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    VPUIP::ArchKind _arch = VPUIP::ArchKind::KMB;
    VPUIP::CompilationMode _compilationMode = VPUIP::CompilationMode::ReferenceSW;
};

SetCompileParamsPass::SetCompileParamsPass(VPUIP::ArchKind arch, VPUIP::CompilationMode compilationMode, Logger log)
        : _arch(arch), _compilationMode(compilationMode) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult SetCompileParamsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    auto archStr = VPUIP::symbolizeEnum<VPUIP::ArchKind>(archOpt.getValue());
    VPUX_THROW_UNLESS(archStr.hasValue(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
    _arch = archStr.getValue();

    auto compilationModeStr = VPUIP::symbolizeEnum<VPUIP::CompilationMode>(compilationModeOpt.getValue());
    VPUX_THROW_UNLESS(compilationModeStr.hasValue(), "Unknown compilation mode: '{0}'", compilationModeOpt.getValue());
    _compilationMode = compilationModeStr.getValue();

    return mlir::success();
}

void SetCompileParamsPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    _log.trace("Set compilation mode to {0}", _compilationMode);
    VPUIP::setCompilationMode(module, _compilationMode);

    _log.trace("Set VPU architecture to {0}", _arch);
    VPUIP::setArch(module, _arch);

    _log.trace("Add VPUIP.Graph Operation");

    const auto options = VPUIP::ExecutionFlagAttr::get(&ctx, VPUIP::ExecutionFlag::NONE);

    const auto version = VPUIP::VersionAttr::get(getInt32Attr(&ctx, MVCNN_VERSION_MAJOR),           // majorV
                                                 getInt32Attr(&ctx, MVCNN_VERSION_MINOR),           // minorV
                                                 getInt32Attr(&ctx, MVCNN_VERSION_PATCH),           // patchV
                                                 mlir::StringAttr::get(&ctx, VPUX_PLUGIN_VERSION),  // hash
                                                 mlir::StringAttr::get(&ctx, "VPUX Compiler"),      // contextStr
                                                 &ctx);

    _log.info("Blob version: majorV={0}, minorV={1}, patch={2}, hash={3}, context={4}", version.majorV().getValue(),
              version.minorV().getValue(), version.patchV().getValue(), version.hash(), version.contextStr());

    OpBuilderLogger builderLog(_log);
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    builder.create<VPUIP::GraphOp>(mlir::UnknownLoc::get(&ctx), options, version);
}

}  // namespace

//
// createSetCompileParamsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetCompileParamsPass() {
    return std::make_unique<SetCompileParamsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetCompileParamsPass(ArchKind arch, CompilationMode compilationMode,
                                                                    Logger log) {
    return std::make_unique<SetCompileParamsPass>(arch, compilationMode, log);
}

//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

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
    VPUIP::ArchKind _arch = VPUIP::ArchKind::VPU3700;
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

    _log.trace("Set VPU architecture to {0}", _arch);
    VPUIP::setArch(module, _arch);

    _log.trace("Set compilation mode to {0}", _compilationMode);
    VPUIP::setCompilationMode(module, _compilationMode);

    _log.trace("Add VPUIP.Graph Operation");

    const auto options = VPUIP::ExecutionFlagAttr::get(&ctx, VPUIP::ExecutionFlag::NONE);

    const auto version = VPUIP::VersionAttr::get(getInt32Attr(&ctx, 3),                         // majorV
                                                 getInt32Attr(&ctx, 11),                        // minorV
                                                 getInt32Attr(&ctx, 0),                         // patchV
                                                 mlir::StringAttr::get(&ctx, ""),               // hash
                                                 mlir::StringAttr::get(&ctx, "VPUX Compiler"),  // contextStr
                                                 &ctx);

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

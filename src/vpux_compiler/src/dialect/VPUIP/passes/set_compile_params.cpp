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
    SetCompileParamsPass(Optional<VPUIP::ArchKind> arch, Optional<VPUIP::CompilationMode> compilationMode, Logger log);

private:
    void safeRunOnModule() final;

private:
    VPUIP::ArchKind _arch;
    VPUIP::CompilationMode _compilationMode;
};

SetCompileParamsPass::SetCompileParamsPass(Optional<VPUIP::ArchKind> arch,
                                           Optional<VPUIP::CompilationMode> compilationMode, Logger log) {
    Base::initLogger(log, Base::getArgumentName());

    if (compilationMode.hasValue()) {
        _compilationMode = compilationMode.getValue();
    } else {
        auto parsed = VPUIP::symbolizeEnum<VPUIP::CompilationMode>(compilationModeName.getValue());
        VPUX_THROW_UNLESS(parsed.hasValue(), "Unknown compilation mode: '{0}'", compilationModeName.getValue());
        _compilationMode = parsed.getValue();
    }

    if (arch.hasValue()) {
        _arch = arch.getValue();
    } else {
        auto parsed = VPUIP::symbolizeEnum<VPUIP::ArchKind>(archName.getValue());
        VPUX_THROW_UNLESS(parsed.hasValue(), "Unknown VPU architecture : '{0}'", archName.getValue());

        _arch = parsed.getValue();
    }
}

void SetCompileParamsPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    _log.trace("Set VPU architecture to {0}", _arch);

    VPUIP::setArch(module, _arch);

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

    VPUIP::setCompilationMode(module, _compilationMode);
}

}  // namespace

//
// createSetCompileParamsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetCompileParamsPass(Optional<ArchKind> arch,
                                                                    Optional<CompilationMode> compilationMode,
                                                                    Logger log) {
    return std::make_unique<SetCompileParamsPass>(arch, compilationMode, log);
}

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

#include "vpux/compiler/dialect/VPUIPRegMapped/passes.hpp"

#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/enums.hpp"
// Alex: #include "vpux/compiler/dialect/VPUIPRegMapped/generated/schema/gf_version.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <version.hpp>

using namespace vpux;

namespace {

//
// SetCompileParamsPass
//

class SetCompileParamsPass final : public VPUIPRegMapped::SetCompileParamsBase<SetCompileParamsPass> {
public:
    SetCompileParamsPass() = default;
    SetCompileParamsPass(VPUIPRegMapped::ArchKind arch, VPUIPRegMapped::CompilationMode compilationMode,
                         Optional<int> numOfDPUGroups, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    VPUIPRegMapped::ArchKind _arch = VPUIPRegMapped::ArchKind::KMB;
    VPUIPRegMapped::CompilationMode _compilationMode = VPUIPRegMapped::CompilationMode::ReferenceSW;
    Optional<int> _numOfDPUGroups;
};

SetCompileParamsPass::SetCompileParamsPass(VPUIPRegMapped::ArchKind arch,
                                           VPUIPRegMapped::CompilationMode compilationMode,
                                           Optional<int> numOfDPUGroups, Logger log)
        : _arch(arch), _compilationMode(compilationMode), _numOfDPUGroups(numOfDPUGroups) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult SetCompileParamsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    auto archStr = VPUIPRegMapped::symbolizeEnum<VPUIPRegMapped::ArchKind>(archOpt.getValue());
    VPUX_THROW_UNLESS(archStr.hasValue(), "Unknown VPU architecture : '{0}'", archOpt.getValue());
    _arch = archStr.getValue();

    auto compilationModeStr =
            VPUIPRegMapped::symbolizeEnum<VPUIPRegMapped::CompilationMode>(compilationModeOpt.getValue());
    VPUX_THROW_UNLESS(compilationModeStr.hasValue(), "Unknown compilation mode: '{0}'", compilationModeOpt.getValue());
    _compilationMode = compilationModeStr.getValue();

    if (numberOfDPUGroupsOpt.hasValue()) {
        _numOfDPUGroups = numberOfDPUGroupsOpt.getValue();
    }

    return mlir::success();
}

void SetCompileParamsPass::safeRunOnModule() {
    // Alex: auto& ctx = getContext();
    auto module = getOperation();

    _log.trace("Set compilation mode to {0}", _compilationMode);
    VPUIPRegMapped::setCompilationMode(module, _compilationMode);

    _log.trace("Set VPU architecture to {0}", _arch);
    // VPUIPRegMapped::setArch(module, _arch, _numOfDPUGroups); // TODO

    _log.trace("Add VPUIPRegMapped.Graph Operation");

    // Alex: const auto options = VPUIPRegMapped::ExecutionFlagAttr::get(&ctx, VPUIPRegMapped::ExecutionFlag::NONE);

    /*
    // Alex
    const auto version = VPUIPRegMapped::VersionAttr::get(getIntAttr(&ctx, MVCNN_VERSION_MAJOR),             // majorV
                                                          getIntAttr(&ctx, MVCNN_VERSION_MINOR),             // minorV
                                                          getIntAttr(&ctx, MVCNN_VERSION_PATCH),             // patchV
                                                          mlir::StringAttr::get(&ctx, VPUX_PLUGIN_VERSION),  // hash
                                                          mlir::StringAttr::get(&ctx, "VPUX Compiler"),  // contextStr
                                                          &ctx);

    _log.info("Blob version: majorV={0}, minorV={1}, patch={2}, hash={3}, context={4}", version.majorV().getValue(),
              version.minorV().getValue(), version.patchV().getValue(), version.hash(), version.contextStr());

    OpBuilderLogger builderLog(_log);
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    builder.create<VPUIPRegMapped::GraphOp>(mlir::UnknownLoc::get(&ctx), options, version);
    */
}

}  // namespace

//
// createSetCompileParamsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIPRegMapped::createSetCompileParamsPass() {
    return std::make_unique<SetCompileParamsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPUIPRegMapped::createSetCompileParamsPass(ArchKind arch,
                                                                             CompilationMode compilationMode,
                                                                             Optional<int> numOfDPUGroups, Logger log) {
    return std::make_unique<SetCompileParamsPass>(arch, compilationMode, numOfDPUGroups, log);
}

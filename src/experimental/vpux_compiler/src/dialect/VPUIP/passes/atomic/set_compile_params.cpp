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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"

using namespace vpux;

namespace {

//
// SetCompileParamsPass
//

class SetCompileParamsPass final : public VPUIP::SetCompileParamsBase<SetCompileParamsPass> {
public:
    SetCompileParamsPass(Optional<VPUIP::ArchKind> arch, Logger log);

public:
    void runOnOperation() final;

private:
    void passBody();

private:
    VPUIP::ArchKind _arch;
    Logger _log;
};

SetCompileParamsPass::SetCompileParamsPass(Optional<VPUIP::ArchKind> arch, Logger log): _log(log) {
    _log.setName(Base::getArgumentName());

    if (arch.hasValue()) {
        _arch = arch.getValue();
    } else {
        auto parsed = VPUIP::symbolizeEnum<VPUIP::ArchKind>(archName.getValue());
        VPUX_THROW_UNLESS(parsed.hasValue(), "Unknown VPU architecture : '{0}'", archName.getValue());

        _arch = parsed.getValue();
    }
}

void SetCompileParamsPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

void SetCompileParamsPass::passBody() {
    auto module = getOperation();
    VPUIP::setArch(module, _arch);
}

}  // namespace

//
// createSetCompileParamsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetCompileParamsPass(Optional<ArchKind> arch, Logger log) {
    return std::make_unique<SetCompileParamsPass>(arch, log);
}

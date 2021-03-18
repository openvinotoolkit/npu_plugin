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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

//
// AdjustForVPUPass
//

class AdjustForVPUPass final : public IE::AdjustForVPUBase<AdjustForVPUPass> {
public:
    explicit AdjustForVPUPass(Logger log);

public:
    void runOnOperation() final;

private:
    void passBody();

private:
    Logger _log;
    mlir::OpPassManager _pm;
};

AdjustForVPUPass::AdjustForVPUPass(Logger log)
        : _log(log), _pm(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _pm.addPass(IE::createConvertFCToConvPass(_log.nest()));
    _pm.addPass(IE::createConvertTile2PerAxisTilePass(_log.nest()));
    _pm.addPass(IE::createConvertPrecisionToFP16Pass(_log.nest()));
    _pm.addPass(IE::createConvertPaddingsToFloorModePass(_log.nest()));
}

void AdjustForVPUPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// passBody
//

void AdjustForVPUPass::passBody() {
    auto module = getOperation();
    if (mlir::failed(runPipeline(_pm, module))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustForVPUPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustForVPUPass(Logger log) {
    return std::make_unique<AdjustForVPUPass>(log);
}

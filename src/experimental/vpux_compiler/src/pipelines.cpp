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

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;

//
// ReferenceMode
//

namespace {

class ReferenceModePass final : public ReferenceModeBase<ReferenceModePass> {
public:
    explicit ReferenceModePass(Logger log);

public:
    void runOnOperation() final;

private:
    Logger _log;
    mlir::OpPassManager _pm;
};

ReferenceModePass::ReferenceModePass(Logger log)
        : _log(log), _pm(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _pm.addPass(IE::createAdjustPrecisionForVPUPass(_log.nest()));
    _pm.addPass(createLowerIE2IERTPass(_log.nest()));
    _pm.addPass(createLowerIERT2VPUIPPass(_log.nest()));
    _pm.addPass(VPUIP::createRemoveExtraDMAPass(_log.nest()));
    _pm.addPass(VPUIP::createAssignTensorOffsetsDDRPass(_log.nest()));
    _pm.addPass(VPUIP::createAddLinearSchedulingPass(_log.nest()));
}

void ReferenceModePass::runOnOperation() {
    if (mlir::failed(runPipeline(_pm, getOperation()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createReferenceModePass
//

std::unique_ptr<mlir::Pass> vpux::createReferenceModePass(Logger log) {
    return std::make_unique<ReferenceModePass>(log);
}

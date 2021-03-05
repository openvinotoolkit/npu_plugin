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

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

//
// LowPrecisionPass
//

class LowPrecisionPass final : public IE::LowPrecisionBase<LowPrecisionPass> {
public:
    explicit LowPrecisionPass(Logger log);

public:
    void runOnFunction() final;

private:
    void passBody();

private:
    Logger _log;
    mlir::OpPassManager _pm;
};

LowPrecisionPass::LowPrecisionPass(Logger log)
        : _log(log), _pm(mlir::FuncOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _pm.addPass(IE::createSplitFakeQuantPass(_log.nest()));
    _pm.addPass(IE::createQuantizeConstPass(_log.nest()));
    // TODO: insert advanced LPT pipeline here
    _pm.addPass(IE::createDequantizeConstPass(_log.nest()));
    _pm.addPass(IE::createMergeFakeQuantPass(_log.nest()));
}

void LowPrecisionPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// passBody
//

void LowPrecisionPass::passBody() {
    auto func = getFunction();
    if (mlir::failed(runPipeline(_pm, func))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowPrecisionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLowPrecisionPass(Logger log) {
    return std::make_unique<LowPrecisionPass>(log);
}

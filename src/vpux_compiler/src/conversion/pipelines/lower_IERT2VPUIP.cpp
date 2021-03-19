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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

//
// LowerIERT2VPUIPPass
//

class LowerIERT2VPUIPPass final : public LowerIERT2VPUIPBase<LowerIERT2VPUIPPass> {
public:
    explicit LowerIERT2VPUIPPass(Logger log);

public:
    void runOnOperation() final;

private:
    void passBody();

    mlir::LogicalResult setRunTimeResources();

private:
    Logger _log;
    mlir::OpPassManager _pm;
};

LowerIERT2VPUIPPass::LowerIERT2VPUIPPass(Logger log)
        : _log(log), _pm(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());
}

void LowerIERT2VPUIPPass::runOnOperation() {
    try {
        auto& ctx = getContext();

        const auto ddrMemSpace = VPUIP::PhysicalMemoryAttr::get(&ctx, VPUIP::PhysicalMemory::DDR);

        _pm.addPass(IERT::createSetInternalMemorySpacePass(ddrMemSpace, _log.nest()));
        _pm.addPass(IERT::createStaticAllocationPass(ddrMemSpace, _log.nest()));
        _pm.addPass(createConvertIERT2VPUIPPass(_log.nest()));

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// passBody
//

void LowerIERT2VPUIPPass::passBody() {
    auto module = getOperation();

    if (mlir::failed(runPipeline(_pm, module))) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(setRunTimeResources())) {
        signalPassFailure();
        return;
    }
}

//
// setRunTimeResources
//

mlir::LogicalResult LowerIERT2VPUIPPass::setRunTimeResources() {
    _log.trace("Setup used run-time resources for executors");

    auto& ctx = getContext();
    auto module = getOperation();

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    if (resources == nullptr) {
        return errorAt(module, "Failed to get 'IERT.RunTimeResources' Operation from Module");
    }

    const auto getProcAttr = [&](VPUIP::PhysicalProcessor proc) {
        return VPUIP::PhysicalProcessorAttr::get(&ctx, proc);
    };

    if (auto available = resources.getAvailableExecutor(getProcAttr(VPUIP::PhysicalProcessor::SHAVE_UPA))) {
        resources.setUsedExecutor(getProcAttr(VPUIP::PhysicalProcessor::SHAVE_UPA), available.count());
    }
    if (auto available = resources.getAvailableExecutor(getProcAttr(VPUIP::PhysicalProcessor::NCE_Cluster))) {
        // We have to set at least 1 NCE cluster to allow run-time work, even for full SW mode.
        resources.setUsedExecutor(getProcAttr(VPUIP::PhysicalProcessor::NCE_Cluster), 1);
    }

    return mlir::success();
}

}  // namespace

//
// createLowerIERT2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createLowerIERT2VPUIPPass(Logger log) {
    return std::make_unique<LowerIERT2VPUIPPass>(log);
}

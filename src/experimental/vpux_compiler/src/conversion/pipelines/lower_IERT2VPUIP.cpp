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

    mlir::LogicalResult replaceCnnNetworkOp();
    mlir::LogicalResult removeGlobalMemRefOp();

private:
    Logger _log;
    mlir::OpPassManager _pm;
};

LowerIERT2VPUIPPass::LowerIERT2VPUIPPass(Logger log)
        : _log(log), _pm(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _pm.addPass(createConvertIERT2VPUIPPass(_log.nest()));
}

void LowerIERT2VPUIPPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} Pass failed : {1}", getName(), e.what());
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

    if (mlir::failed(replaceCnnNetworkOp())) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(removeGlobalMemRefOp())) {
        signalPassFailure();
        return;
    }
}

//
// replaceCnnNetworkOp
//

mlir::LogicalResult LowerIERT2VPUIPPass::replaceCnnNetworkOp() {
    _log.trace("Replace IERT.CNNNetwork Operation with VPUIP.Graph");

    auto& ctx = getContext();
    auto module = getOperation();

    IERT::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    if (mlir::failed(IERT::CNNNetworkOp::getFromModule(module, netOp, netFunc))) {
        return printTo(module.emitError(), "Failed to get IERT.CNNNetwork Operation from module");
    }

    auto options = VPUIP::ExecutionFlagAttr::get(VPUIP::ExecutionFlag::NONE, &ctx);

    const auto getProcAttr = [&](VPUIP::PhysicalProcessor proc) {
        return VPUIP::PhysicalProcessorAttr::get(proc, &ctx);
    };

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    if (resources == nullptr) {
        return printTo(module.emitError(), "Failed to get IERT.RunTimeResources Operation from module");
    }
    if (auto available = resources.getAvailableExecutor(getProcAttr(VPUIP::PhysicalProcessor::SHAVE_UPA))) {
        resources.setUsedExecutor(getProcAttr(VPUIP::PhysicalProcessor::SHAVE_UPA), available.count());
    }
    if (auto available = resources.getAvailableExecutor(getProcAttr(VPUIP::PhysicalProcessor::NCE_Cluster))) {
        resources.setUsedExecutor(getProcAttr(VPUIP::PhysicalProcessor::NCE_Cluster), 1);
    }

    auto version = VPUIP::VersionAttr::get(getInt32Attr(&ctx, 3),                         // majorV
                                           getInt32Attr(&ctx, 11),                        // minorV
                                           getInt32Attr(&ctx, 0),                         // patchV
                                           mlir::StringAttr::get("", &ctx),               // hash
                                           mlir::StringAttr::get("VPUX Compiler", &ctx),  // contextStr
                                           &ctx);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);

    auto graphOp = builder.create<VPUIP::GraphOp>(netOp.getLoc(), netOp.netNameAttr(), netOp.entryPointAttr(), options,
                                                  version);
    VPUIP::GraphOp::ensureTerminator(graphOp.inputsInfo(), builder, netOp.getLoc());
    VPUIP::GraphOp::ensureTerminator(graphOp.outputsInfo(), builder, netOp.getLoc());

    builder.setInsertionPointToStart(&graphOp.inputsInfo().front());
    for (auto dataInfo : netOp.inputsInfo().getOps<IERT::DataInfoOp>()) {
        builder.create<VPUIP::TensorInfoOp>(dataInfo.getLoc(), dataInfo.nameAttr(), dataInfo.precisionAttr(),
                                            dataInfo.layoutAttr());
    }

    builder.setInsertionPointToStart(&graphOp.outputsInfo().front());
    for (auto dataInfo : netOp.outputsInfo().getOps<IERT::DataInfoOp>()) {
        builder.create<VPUIP::TensorInfoOp>(dataInfo.getLoc(), dataInfo.nameAttr(), dataInfo.precisionAttr(),
                                            dataInfo.layoutAttr());
    }

    netOp.erase();

    return mlir::success();
}

//
// removeGlobalMemRefOp
//

mlir::LogicalResult LowerIERT2VPUIPPass::removeGlobalMemRefOp() {
    _log.trace("Remove GlobalMemRef Operations from module");

    auto module = getOperation();

    auto callback = [&](mlir::GlobalMemrefOp op) -> mlir::WalkResult {
        auto uses = op.getSymbolUses(module);
        if (uses.hasValue() && !uses->empty()) {
            _log.error("GlobalMemrefOp Operation '{0}' still has uses in IR", op);
            return mlir::failure();
        }

        op.erase();

        return mlir::success();
    };

    return mlir::success(!module.walk(callback).wasInterrupted());
}

}  // namespace

//
// createLowerIERT2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createLowerIERT2VPUIPPass(Logger log) {
    return std::make_unique<LowerIERT2VPUIPPass>(log);
}

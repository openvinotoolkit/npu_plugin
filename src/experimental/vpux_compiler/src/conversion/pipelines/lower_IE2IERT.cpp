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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

//
// LowerIE2IERTPass
//

class LowerIE2IERTPass final : public LowerIE2IERTBase<LowerIE2IERTPass> {
public:
    explicit LowerIE2IERTPass(Logger log);

public:
    void runOnOperation() final;

private:
    struct DataInfo final {
        mlir::StringAttr name;
        mlir::TypeAttr precision;
        mlir::AffineMapAttr layout;
    };

private:
    void passBody();

    mlir::LogicalResult removeCnnNetworkOp();
    mlir::LogicalResult restoreCnnNetworkOp();

private:
    Logger _log;
    mlir::OpPassManager _pm;
    mlir::LocationAttr _netInfoLoc;
    mlir::StringAttr _netName;
    mlir::FlatSymbolRefAttr _entryPoint;
    SmallVector<DataInfo, 1> _inputsInfo;
    SmallVector<DataInfo, 1> _outputsInfo;
};

LowerIE2IERTPass::LowerIE2IERTPass(Logger log)
        : _log(log), _pm(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _pm.addPass(createBufferizeIEPass(_log.nest()));
    _pm.addPass(mlir::createTensorConstantBufferizePass());
    _pm.addPass(mlir::createFuncBufferizePass());
    _pm.addPass(mlir::createBufferResultsToOutParamsPass());
    _pm.addPass(mlir::createFinalizingBufferizePass());
    _pm.addPass(mlir::createBufferDeallocationPass());
    _pm.addPass(mlir::createCopyRemovalPass());
}

void LowerIE2IERTPass::runOnOperation() {
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

void LowerIE2IERTPass::passBody() {
    auto module = getOperation();

    if (mlir::failed(removeCnnNetworkOp())) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(runPipeline(_pm, module))) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(restoreCnnNetworkOp())) {
        signalPassFailure();
        return;
    }
}

mlir::LogicalResult LowerIE2IERTPass::removeCnnNetworkOp() {
    _log.trace("Remove IE::CNNNetwork Operation from IR");

    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    if (mlir::failed(IE::CNNNetworkOp::getFromModule(module, netOp, netFunc))) {
        _log.error("Failed to get IE::CNNNetwork Operation from module");
        return mlir::failure();
    }

    _netInfoLoc = netOp.getLoc();
    _netName = netOp.netNameAttr();
    _entryPoint = netOp.entryPointAttr();

    for (auto dataInfo : netOp.inputsInfo().getOps<IE::DataInfoOp>()) {
        _inputsInfo.push_back(DataInfo{dataInfo.nameAttr(), dataInfo.precisionAttr(),
                                       mlir::AffineMapAttr::get(getAffineMap(module.getContext(), dataInfo.layout()))});
    }
    for (auto dataInfo : netOp.outputsInfo().getOps<IE::DataInfoOp>()) {
        _outputsInfo.push_back(
                DataInfo{dataInfo.nameAttr(), dataInfo.precisionAttr(),
                         mlir::AffineMapAttr::get(getAffineMap(module.getContext(), dataInfo.layout()))});
    }

    netOp.erase();

    return mlir::success();
}

mlir::LogicalResult LowerIE2IERTPass::restoreCnnNetworkOp() {
    _log.trace("Restore IERT::CNNNetwork Operation");

    auto module = getOperation();

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());

    auto netOp = builder.create<IERT::CNNNetworkOp>(_netInfoLoc, _netName, _entryPoint);
    IERT::CNNNetworkOp::ensureTerminator(netOp.inputsInfo(), builder, _netInfoLoc);
    IERT::CNNNetworkOp::ensureTerminator(netOp.outputsInfo(), builder, _netInfoLoc);

    builder.setInsertionPointToStart(&netOp.inputsInfo().front());
    for (const auto& info : _inputsInfo) {
        builder.create<IERT::DataInfoOp>(_netInfoLoc, info.name, info.precision, info.layout);
    }

    builder.setInsertionPointToStart(&netOp.outputsInfo().front());
    for (const auto& info : _outputsInfo) {
        builder.create<IERT::DataInfoOp>(_netInfoLoc, info.name, info.precision, info.layout);
    }

    return mlir::success();
}

}  // namespace

//
// createLowerIE2IERTPass
//

std::unique_ptr<mlir::Pass> vpux::createLowerIE2IERTPass(Logger log) {
    return std::make_unique<LowerIE2IERTPass>(log);
}

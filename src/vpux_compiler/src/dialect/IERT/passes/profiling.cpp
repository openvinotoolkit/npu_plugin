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

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/IR/Attributes.h"

using namespace vpux;

namespace {

//
// TimestampProfilingPass
//

class TimestampProfilingPass final : public IERT::TimestampProfilingBase<TimestampProfilingPass> {
public:
    explicit TimestampProfilingPass(IERT::AttrCreateFunc memSpaceCb, Logger log): _memSpaceCb(std::move(memSpaceCb)) {
        VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

//
// DMATaskProfilingPass
//

class DMATaskProfilingPass final : public IERT::DMATaskProfilingBase<DMATaskProfilingPass> {
public:
    explicit DMATaskProfilingPass(IERT::AttrCreateFunc memSpaceCb, Logger log): _memSpaceCb(std::move(memSpaceCb)) {
        VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

//
// DPUProfilingPass
//

class DPUProfilingPass final : public IERT::DPUProfilingBase<DPUProfilingPass> {
public:
    explicit DPUProfilingPass(IERT::AttrCreateFunc memSpaceCb, Logger log): _memSpaceCb(std::move(memSpaceCb)) {
        VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

//
// safeRunOnModule
//

void TimestampProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    _memSpace = _memSpaceCb(ctx, "");
    if (_memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    SmallVector<std::pair<IERT::AsyncLayerOpInterface, VPU::ExecutorKindAttr>> layerTasks;

    netFunc.walk([&](IERT::AsyncLayerOpInterface curTask) {
        uint32_t curNumUnits = 0;
        const auto curExecutor = curTask.getExecutor(curNumUnits);

        auto physType = curExecutor.dyn_cast<VPU::ExecutorKindAttr>();
        if (physType == nullptr) {
            _log.trace("It is not a ExecutorKind Task");
            return;
        }

        layerTasks.push_back({curTask, physType});
    });

    VPUX_THROW_WHEN(layerTasks.empty(), "No TimestampOp was added");

    const auto output_size = static_cast<int64_t>(layerTasks.size());

    const auto timestampType = getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, _memSpace);
    const auto cmxMemType = getMemRefType(ShapeRef({output_size}), getUInt32Type(ctx), DimsOrder::C, _memSpace);
    const auto outputResult = mlir::MemRefType::get({output_size}, getUInt32Type(ctx));

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    auto memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx), cmxMemType);

    SmallVector<mlir::Value> timestamps;
    for (auto layer : layerTasks) {
        auto curTask = layer.first;
        auto physType = layer.second;
        _log.trace("Process Operation '{0}'", curTask->getLoc());

        builder.setInsertionPointAfter(curTask);
        int layerNumber = 0;
        std::string curTaskName = "[";
        curTaskName += curTask->getName().getStringRef().data();
        if (physType.getValue() == VPU::ExecutorKind::NCE || physType.getValue() == VPU::ExecutorKind::DPU) {
            curTaskName += "_DPU]";
        } else {
            curTaskName += "_NA]";
        }

        curTaskName += stringifyLocation(curTask->getLoc());

        auto name = mlir::NameLoc::get(mlir::Identifier::get(
                curTaskName +
                        ((timestamps.size() == 0) ? "_PROFBEGIN_0"
                                                  : ("_PROFMIDDLE_" + std::to_string(timestamps.size() - 1))) +
                        "_" + std::to_string(layerNumber),
                ctx));
        auto sub = builder.create<IERT::SubViewOp>(mlir::NameLoc::get(mlir::Identifier::get("subview", ctx)), memOp,
                                                   SmallVector<int64_t>({static_cast<int64_t>(timestamps.size())}),
                                                   timestampType.getShape());

        timestamps.push_back(builder.create<IERT::TimestampOp>(name, timestampType, sub).output());
    }

    auto concatview = builder.create<IERT::ConcatViewOp>(mlir::NameLoc::get(mlir::Identifier::get("concatview", ctx)),
                                                         timestamps, memOp.memref());

    auto profilngResult = AddNewProfilingOutput(ctx, netFunc, netOp, outputResult, "profilingOutput");
    auto copyLoc2 = mlir::NameLoc::get(mlir::Identifier::get("ProfilingCMX2DDR", ctx));
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc2, concatview.output(), profilngResult);

    // Add result to the returnOp
    netFunc.walk([&](mlir::ReturnOp op) {
        op.operandsMutable().append(outputOp.output());
    });
}

void DMATaskProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    _memSpace = _memSpaceCb(ctx, "");
    if (_memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<mlir::async::ExecuteOp> executeOps;
    auto timestampType = getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, _memSpace);

    netFunc.walk([&](mlir::async::ExecuteOp execOp) {
        _log.trace("Process Operation '{0}'", execOp->getLoc());

        bool found = false;
        auto& bodyBlock = execOp.body().front();
        bodyBlock.walk([&](IERT::CopyOp curTask) {
            auto curTaskName = stringifyLocation(curTask->getLoc());
            if (curTaskName.find("ProfilingCMX2DDR") == std::string::npos) {
                found = true;
            }
        });
        if (found) {
            executeOps.push_back(execOp);
        }
    });

    VPUX_THROW_UNLESS(executeOps.size(), "No TimestampOp was added");

    // For each measured DMA operations two timestamps will be captured
    int output_size = executeOps.size() * 2;
    auto cmxMemType = getMemRefType(ShapeRef({output_size}), getUInt32Type(ctx), DimsOrder::C, _memSpace);
    auto outputResult = mlir::MemRefType::get({output_size}, getUInt32Type(ctx));

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    auto memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx), cmxMemType);

    SmallVector<mlir::Value> timestampsOps;
    unsigned dma_id = 0;
    for (auto& execOp : executeOps) {
        mlir::Operation* firstCopy = nullptr;
        mlir::Operation* lastCopy = nullptr;
        auto& bodyBlock = execOp.body().front();
        bodyBlock.walk([&](IERT::CopyOp curTask) {
            lastCopy = curTask.getOperation();
            if (firstCopy == nullptr)
                firstCopy = lastCopy;
        });

        auto insertDma = [&](mlir::Operation* op, bool after) {
            if (after) {
                builder.setInsertionPointAfter(op);
            } else {
                builder.setInsertionPoint(op);
            }
            auto sub = builder.create<IERT::SubViewOp>(
                    mlir::NameLoc::get(mlir::Identifier::get("dmaProfilingSubview", ctx)), memOp,
                    SmallVector<int64_t>({static_cast<int64_t>(dma_id)}), timestampType.getShape());
            std::string curTaskName;
            curTaskName = stringifyLocation(op->getLoc());
            auto name = mlir::NameLoc::get(mlir::Identifier::get(
                    curTaskName + ((!after) ? (timestampsOps.size() == 0 ? "_PROFBEGIN" : "_PROFTASKBEGIN")
                                            : ("_PROFTASKEND_" + std::to_string(dma_id - 1) + "_" +
                                               std::to_string(dma_id / 2 + 1))),
                    ctx));
            dma_id++;
            return builder.create<IERT::TimestampOp>(name, timestampType, sub).output();
        };
        SmallVector<mlir::Value> localTimestampsOps;
        localTimestampsOps.push_back(insertDma(firstCopy, false));
        localTimestampsOps.push_back(insertDma(lastCopy, true));
        auto yieldOp = mlir::dyn_cast<mlir::async::YieldOp>(execOp.body().front().getTerminator());
        unsigned firstTimestampOperandId = yieldOp.operands().size();
        yieldOp.operandsMutable().append(localTimestampsOps);

        auto* bodyBlockPtr = &execOp.body().front();
        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
            mlir::BlockAndValueMapping mapper;

            const auto curBlockArgs = bodyBlockPtr->getArguments();
            for (size_t i = 0; i < blockArgs.size(); ++i) {
                mapper.map(curBlockArgs[i], blockArgs[i]);
            }

            SmallVector<mlir::Value> newResults;
            for (auto& op : bodyBlock.getOperations()) {
                if (!mlir::isa<mlir::async::YieldOp>(op)) {
                    builder.clone(op, mapper);
                } else {
                    for (auto operand : op.getOperands()) {
                        newResults.push_back(mapper.lookupOrDefault(operand));
                    }
                }
            }
            builder.create<mlir::async::YieldOp>(loc, newResults);
        };
        builder.setInsertionPointAfter(execOp);
        auto newExecOp = builder.create<mlir::async::ExecuteOp>(execOp->getLoc(), yieldOp->getOperandTypes(),
                                                                execOp.dependencies(), execOp.operands(), bodyBuilder);
        uint32_t numExecutorUnits = 0;
        auto executor = vpux::IERT::IERTDialect::getExecutor(execOp, numExecutorUnits);
        IERT::IERTDialect::setExecutor(newExecOp, executor, numExecutorUnits);

        for (size_t id = 0; id < localTimestampsOps.size(); id++) {
            timestampsOps.push_back(newExecOp.results()[firstTimestampOperandId + id]);
        }

        execOp->replaceAllUsesWith(newExecOp);
        execOp->erase();
    }

    // Declare and create additional output from network
    auto profilngResult = AddNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dma");

    // Add ExecuteOp with Copy from CMX to DDR
    auto copyLoc = mlir::NameLoc::get(mlir::Identifier::get("profilingCMX2DDR", ctx));
    builder.setInsertionPoint(netFunc.getBody().front().getTerminator());
    auto execOp = builder.create<mlir::async::ExecuteOp>(copyLoc, outputResult, None, None);

    SmallVector<mlir::Value> values;
    for (auto value : timestampsOps) {
        execOp.operandsMutable().append(value);
        auto asyncType = value.getType().dyn_cast<mlir::async::ValueType>();
        values.push_back(execOp.getBody()->addArgument(asyncType.getValueType()));
    }
    auto bodyBlock = &execOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);
    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("dmaProfilingConcat", ctx)), values, memOp.memref());
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc, concatview.output(), profilngResult);
    builder.create<mlir::async::YieldOp>(copyLoc, outputOp->getResults());

    // Add execution attributes to async exec op
    uint32_t numExecutorUnits = 0;
    auto newOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(outputOp.getOperation());
    auto executor = newOpExecutor.getExecutor(numExecutorUnits);
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(execOp, executor, numExecutorUnits);
    }

    builder.setInsertionPointAfter(execOp);
    auto waitOp = builder.create<mlir::async::AwaitOp>(execOp->getLoc(), execOp.results()[0]);

    // And new result(outputBuffer) to the returnOp
    netFunc.walk([&](mlir::ReturnOp op) {
        op.operandsMutable().append(waitOp.result());
    });

    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);
    depsInfo.updateTokenDependencies();
}

void DPUProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    _memSpace = _memSpaceCb(ctx, "");
    if (_memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<std::pair<VPUIP::NCEClusterTaskOp, unsigned>> dpuTasks;
    netFunc.walk([&](VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
        _log.trace("Process Operation '{0}'", nceClusterTaskOp->getLoc());

        auto dpuIt = nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>();
        auto count = std::distance(dpuIt.begin(), dpuIt.end());
        dpuTasks.push_back({nceClusterTaskOp, count});
    });

    VPUX_THROW_UNLESS(dpuTasks.size(), "No TimestampOp was added");

    unsigned elementSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES / sizeof(uint64_t);
    unsigned output_size = dpuTasks.size() * elementSize;
    auto cmxMemType = getMemRefType({output_size}, getUInt64Type(ctx), DimsOrder::C, _memSpace);
    auto outputResult = mlir::MemRefType::get({output_size}, getUInt64Type(ctx));

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    auto memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx), cmxMemType);

    unsigned dpu_id = 0;
    SmallVector<mlir::Value> dpuProfilingOutputs;
    for (auto& dpuTask : dpuTasks) {
        auto cluster = dpuTask.first;
        builder.setInsertionPointAfter(cluster);
        auto timestampType = getMemRefType({elementSize}, getUInt64Type(ctx), DimsOrder::C, _memSpace);
        auto sub = builder.create<IERT::SubViewOp>(
                mlir::NameLoc::get(mlir::Identifier::get("dpuProfilingSubview", ctx)), memOp,
                SmallVector<int64_t>({static_cast<int>(dpu_id * elementSize)}), timestampType.getShape());

        SmallVector<mlir::Type> newResultTypes(cluster.getResultTypes());
        newResultTypes.push_back(timestampType);
        auto newCluster = builder.create<VPUIP::NCEClusterTaskOp>(cluster.getLoc(), newResultTypes,
                                                                  cluster->getOperands(), cluster->getAttrs());

        for (const auto region : llvm::enumerate(cluster.getRegions())) {
            newCluster.getRegion(region.index()).takeBody(*region.value());
        }
        newCluster.profiling_dataMutable().assign(sub);
        dpuProfilingOutputs.push_back(newCluster.profiling_output());

        cluster->replaceAllUsesWith(newCluster);
        cluster->erase();
        dpu_id++;
    }

    VPUX_THROW_UNLESS(dpuProfilingOutputs.size(), "No DPU profiling outputs was added");

    // Declare and create additional output from network
    auto profilngResult = AddNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dpu");

    // Create DMA from CMX to Profiling Output
    auto copyLoc2 = mlir::NameLoc::get(mlir::Identifier::get("dpuProfilingCMX2DDR", ctx));
    builder.setInsertionPoint(netFunc.getBody().front().getTerminator());
    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("dpuProfilingConcat", ctx)), dpuProfilingOutputs, memOp.memref());
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc2, concatview.output(), profilngResult);

    // Add result to the returnOp
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    returnOp.operandsMutable().append(outputOp.output());
}

}  // namespace

//
// createTimestampProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createTimestampProfilingPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<TimestampProfilingPass>(std::move(memSpaceCb), log);
}

//
// createDMATaskProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createDMATaskProfilingPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<DMATaskProfilingPass>(std::move(memSpaceCb), log);
}

//
// createDPUProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createDPUProfilingPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<DPUProfilingPass>(std::move(memSpaceCb), log);
}

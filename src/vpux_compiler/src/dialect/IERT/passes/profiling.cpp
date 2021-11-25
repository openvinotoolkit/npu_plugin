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

mlir::Value AddCMX2DDRExecuteOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::BlockArgument& profilingResult,
                                mlir::Value cmxMemOp, SmallVector<mlir::Value>& timestampsOps, unsigned elementSize,
                                unsigned offset, StringRef name) {
    const auto resultType =
            mlir::MemRefType::get({static_cast<int64_t>(timestampsOps.size() * elementSize)}, getUInt32Type(ctx));

    // Add ExecuteOp with Copy from CMX to DDR
    auto copyLoc = mlir::NameLoc::get(mlir::Identifier::get(name + "ProfilingCMX2DDR" + std::to_string(offset), ctx));
    auto execOp = builder.create<mlir::async::ExecuteOp>(copyLoc, resultType, None, None);

    SmallVector<mlir::Value> values;
    for (auto value : timestampsOps) {
        execOp.operandsMutable().append(value);
        auto asyncType = value.getType().dyn_cast<mlir::async::ValueType>();
        values.push_back(execOp.getBody()->addArgument(asyncType.getValueType()));
    }
    auto bodyBlock = &execOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);
    auto sub = builder.create<IERT::SubViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "DDR" + std::to_string(offset), ctx)), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset * elementSize)}), resultType.getShape());
    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "Profiling" + std::to_string(offset), ctx)), values,
            cmxMemOp);
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc, concatview.output(), sub);
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

    timestampsOps.clear();
    return waitOp.result();
};

mlir::Value AddCMX2DDRCopyOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::BlockArgument& profilingResult,
                             mlir::Value cmxMemOp, SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned elementSize,
                             unsigned offset, StringRef name) {
    const auto resultType =
            mlir::MemRefType::get({static_cast<int64_t>(dpuProfilingOutputs.size() * elementSize)}, getUInt64Type(ctx));

    auto sub = builder.create<IERT::SubViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "DDR" + std::to_string(offset), ctx)), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset * elementSize)}), resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc2 = mlir::NameLoc::get(mlir::Identifier::get(name + "ProfilingCMX2DDR" + std::to_string(offset), ctx));
    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "ProfilingConcat" + std::to_string(offset), ctx)),
            dpuProfilingOutputs, cmxMemOp);

    dpuProfilingOutputs.clear();
    return builder.create<IERT::CopyOp>(copyLoc2, concatview.output(), sub).output();
};

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

    // Find all execOp which contains CopyOps
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
    const unsigned output_size = executeOps.size() * 2;

    // Calculate number of chunks and DMA operation inside one chunk
    // based on the maximum CMX buffer size
    const unsigned total_size_bytes = output_size * sizeof(uint32_t);
    const unsigned chunks = ceil((double)total_size_bytes / VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE);
    const unsigned ops_in_chunk = VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE / sizeof(uint32_t);
    unsigned last_chunk = (total_size_bytes % VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE) / sizeof(uint32_t);
    if (!last_chunk)
        last_chunk = ops_in_chunk;
    llvm::errs() << "total_size_bytes=" << total_size_bytes << "\nchunks=" << chunks
                 << "\nops_in_chunk=" << ops_in_chunk << "\nlast_chunk=" << last_chunk << "\n";

    const auto cmxMemType = getMemRefType(ShapeRef({ops_in_chunk}), getUInt32Type(ctx), DimsOrder::C, _memSpace);
    const auto cmxMemTypeLast = getMemRefType(ShapeRef({last_chunk}), getUInt32Type(ctx), DimsOrder::C, _memSpace);
    const auto outputResult = mlir::MemRefType::get({output_size}, getUInt32Type(ctx));

    // Declare and create additional output from network
    auto profilingResult = AddNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dma");

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    mlir::memref::AllocOp memOp;

    unsigned dma_id = 0;                     // Total DMA ops counter
    unsigned chunk_id = 0;                   // Chunk id
    unsigned chunk_dma_id = 0;               // DMA id inside current chunk
    SmallVector<mlir::Value> timestampsOps;  // Collect chunk timestimps(Cleared inside AddCMX2DDRExecuteOp)
    SmallVector<mlir::Value> waitOps;        // Collect chunk results
    for (auto& execOp : executeOps) {
        // Start new chunk once we reached the end of the previous one
        if (chunk_dma_id && ((chunk_dma_id % ops_in_chunk) == 0)) {
            chunk_dma_id = 0;
            chunk_id++;
        }
        // Beginning of the chunk
        // Push previous operations to DDR and allocate new memory in CMX
        if (chunk_dma_id == 0) {
            if (chunk_id) {
                waitOps.push_back(AddCMX2DDRExecuteOp(builder, ctx, profilingResult, memOp, timestampsOps, 1,
                                                      (chunk_id - 1) * ops_in_chunk, "dma"));
            }
            builder.restoreInsertionPoint(lastInsertionPoint);
            memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx),
                                                          (chunk_id != chunks - 1) ? cmxMemType : cmxMemTypeLast);
            lastInsertionPoint = builder.saveInsertionPoint();
        }

        // Walk thought all copyOp inside one async
        // in order to find the first and the last copyOp inside current execOp
        mlir::Operation* firstCopy = nullptr;
        mlir::Operation* lastCopy = nullptr;
        auto& bodyBlock = execOp.body().front();
        bodyBlock.walk([&](IERT::CopyOp curTask) {
            lastCopy = curTask.getOperation();
            if (firstCopy == nullptr)
                firstCopy = lastCopy;
        });

        //
        // Insertion of Timestamp Ops to the current execOp
        //
        auto insertDma = [&](mlir::Operation* op, bool after) {
            if (after) {
                builder.setInsertionPointAfter(op);
            } else {
                builder.setInsertionPoint(op);
            }
            auto sub = builder.create<IERT::SubViewOp>(
                    mlir::NameLoc::get(mlir::Identifier::get("dmaProfilingSubview", ctx)), memOp,
                    SmallVector<int64_t>({static_cast<int64_t>(chunk_dma_id)}), timestampType.getShape());
            std::string curTaskName;
            curTaskName = stringifyLocation(op->getLoc());
            auto name = mlir::NameLoc::get(
                    mlir::Identifier::get(curTaskName + ((!after) ? (dma_id == 0 ? "_PROFBEGIN" : "_PROFTASKBEGIN")
                                                                  : ("_PROFTASKEND_" + std::to_string(dma_id - 1) +
                                                                     "_" + std::to_string(dma_id / 2 + 1))),
                                          ctx));
            dma_id++;
            chunk_dma_id++;
            return builder.create<IERT::TimestampOp>(name, timestampType, sub).output();
        };
        SmallVector<mlir::Value> localTimestampsOps;
        localTimestampsOps.push_back(insertDma(firstCopy, false));
        localTimestampsOps.push_back(insertDma(lastCopy, true));

        // Prepare for execOp rebuilding: Add new results to the current yieldOp
        auto yieldOp = mlir::dyn_cast<mlir::async::YieldOp>(execOp.body().front().getTerminator());
        unsigned firstTimestampOperandId = yieldOp.operands().size();
        yieldOp.operandsMutable().append(localTimestampsOps);

        //
        // Rebuild current execOp in order to add new results
        //
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

        // Remove old execOp
        execOp->replaceAllUsesWith(newExecOp);
        execOp->erase();
    }
    // Copy to DDR the last chunk
    waitOps.push_back(AddCMX2DDRExecuteOp(builder, ctx, profilingResult, memOp, timestampsOps, 1,
                                          chunk_id * ops_in_chunk, "dma"));

    //
    // Concat all chunks together and push to the network returnOp
    //
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("dmaDDRProfiling", ctx)), waitOps, profilingResult);
    returnOp.operandsMutable().append(concatview.output());

    // Recalculate the async Deps Information
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

    const unsigned elementSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES / sizeof(uint64_t);
    const unsigned output_size = dpuTasks.size() * elementSize;

    // Calculate number of chunks and DMA operation inside one chunk
    // based on the maximum CMX buffer size
    const unsigned total_size_bytes = output_size * sizeof(uint64_t);
    const unsigned chunks = ceil((double)total_size_bytes / VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE);
    const unsigned ops_in_chunk = VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE / VPUIP::HW_DPU_PROFILING_SIZE_BYTES;
    unsigned last_chunk =
            (total_size_bytes % VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE) / VPUIP::HW_DPU_PROFILING_SIZE_BYTES;
    if (!last_chunk)
        last_chunk = ops_in_chunk;
    llvm::errs() << "total_size_bytes=" << total_size_bytes << "\nchunks=" << chunks
                 << "\nops_in_chunk=" << ops_in_chunk << "\nlast_chunk=" << last_chunk << "\n";

    const auto cmxMemType =
            getMemRefType(ShapeRef({ops_in_chunk * elementSize}), getUInt64Type(ctx), DimsOrder::C, _memSpace);
    const auto cmxMemTypeLast =
            getMemRefType(ShapeRef({last_chunk * elementSize}), getUInt64Type(ctx), DimsOrder::C, _memSpace);
    const auto outputResult = mlir::MemRefType::get({output_size}, getUInt64Type(ctx));

    // Declare and create additional output from network
    auto profilingResult = AddNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dpu");

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    mlir::memref::AllocOp memOp;

    unsigned chunk_id = 0;      // Chunk id
    unsigned chunk_dpu_id = 0;  // DPU id inside current chunk
    SmallVector<mlir::Value> dpuProfilingOutputs;
    SmallVector<mlir::Value> waitOps;  // Collect chunk results
    for (auto& dpuTask : dpuTasks) {
        // Start new chunk once we reached the end of the previous one
        if (chunk_dpu_id && ((chunk_dpu_id % ops_in_chunk) == 0)) {
            chunk_dpu_id = 0;
            chunk_id++;
        }
        // Beginning of the chunk
        // Push previous operations to DDR and allocate new memory in CMX
        if (chunk_dpu_id == 0) {
            if (chunk_id) {
                waitOps.push_back(AddCMX2DDRCopyOp(builder, ctx, profilingResult, memOp, dpuProfilingOutputs,
                                                   elementSize, (chunk_id - 1) * ops_in_chunk, "dpu"));
            }
            builder.restoreInsertionPoint(lastInsertionPoint);
            memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx),
                                                          (chunk_id != chunks - 1) ? cmxMemType : cmxMemTypeLast);
            lastInsertionPoint = builder.saveInsertionPoint();
        }

        auto cluster = dpuTask.first;
        builder.setInsertionPointAfter(cluster);
        auto timestampType = getMemRefType({elementSize}, getUInt64Type(ctx), DimsOrder::C, _memSpace);
        auto sub = builder.create<IERT::SubViewOp>(
                mlir::NameLoc::get(mlir::Identifier::get("dpuProfilingSubview", ctx)), memOp,
                SmallVector<int64_t>({static_cast<int>(chunk_dpu_id * elementSize)}), timestampType.getShape());

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
        chunk_dpu_id++;
    }
    // Copy to DDR the last chunk
    waitOps.push_back(AddCMX2DDRCopyOp(builder, ctx, profilingResult, memOp, dpuProfilingOutputs, elementSize,
                                       chunk_id * ops_in_chunk, "dpu"));

    //
    // Concat all chunks together and push to the network returnOp
    //
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("dpuDDRProfiling", ctx)), waitOps, profilingResult);
    returnOp.operandsMutable().append(concatview.output());
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

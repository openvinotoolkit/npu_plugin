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
#include "vpux/compiler/utils/rewriter.hpp"
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
};

mlir::Value AddCMX2DDRExecuteOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::BlockArgument& profilingResult,
                                mlir::Value cmxMemOp, SmallVector<mlir::Value>& timestampsOps, unsigned elementSize,
                                unsigned offset, StringRef name) {
    const auto resultType =
            mlir::MemRefType::get({static_cast<int64_t>(timestampsOps.size() * elementSize)}, getUInt32Type(ctx));

    // Add ExecuteOp with Copy from CMX to DDR
    auto copyLoc =
            mlir::NameLoc::get(mlir::Identifier::get(name + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset), ctx));
    auto execOp = builder.create<mlir::async::ExecuteOp>(copyLoc, resultType, None, None);

    SmallVector<mlir::Value> values;
    for (auto value : timestampsOps) {
        execOp.operandsMutable().append(value);
        auto asyncType = value.getType().dyn_cast<mlir::async::ValueType>();
        if (asyncType) {
            values.push_back(execOp.getBody()->addArgument(asyncType.getValueType()));
        }
    }
    auto bodyBlock = &execOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);
    auto sub = builder.create<IERT::SubViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "DDR" + std::to_string(offset), ctx)), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * elementSize}), resultType.getShape());
    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "Profiling" + std::to_string(offset), ctx)), values,
            cmxMemOp);
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc, concatview.output(), sub);
    builder.create<mlir::async::YieldOp>(copyLoc, outputOp->getResults());

    // Add execution attributes to async exec op
    auto newOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(outputOp.getOperation());
    auto executor = newOpExecutor.getExecutor();
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(execOp, executor);
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
            SmallVector<int64_t>({static_cast<int64_t>(offset) * elementSize}), resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc2 =
            mlir::NameLoc::get(mlir::Identifier::get(name + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset), ctx));
    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "ProfilingConcat" + std::to_string(offset), ctx)),
            dpuProfilingOutputs, cmxMemOp);

    dpuProfilingOutputs.clear();
    return builder.create<IERT::CopyOp>(copyLoc2, concatview.output(), sub).output();
};

// DMA profiling pass
// Wraps all DMA operation in the network except for profiling management one with the two
// timestamps DMAs inside one async.execute in order to guarantee no barriers execution
// Steps:
//   1. Allocate buffer in CMX for the first chunk(configured via HW_DMA_PROFILING_MAX_BUFFER_SIZE)
//   2. Fill it with results of timestamp operations
//   3. Connect results to the ConcatOp
//   4. Send result of concatOp to DDR using new CopyOp
//   5. Allocate buffer for the next chunk and continue with steps 2-4
//   6. Connect all DMA to DDR operations to the ConcatOp and connect it to the new network profiling output
void DMATaskProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto memSpace = _memSpaceCb(ctx, "");
    if (memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<mlir::async::ExecuteOp> executeOps;
    auto timestampType = getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, memSpace);

    // Find all execOp which contains CopyOps
    netFunc.walk([&](mlir::async::ExecuteOp execOp) {
        _log.trace("Process Operation '{0}'", execOp->getLoc());

        bool found = false;
        auto& bodyBlock = execOp.body().front();
        bodyBlock.walk([&](IERT::CopyOp curTask) {
            auto curTaskName = stringifyLocation(curTask->getLoc());
            // Skip DMAs which are used for handling profiling data. Such DMAs will not be measured.
            if (curTaskName.find(PROFILING_CMX_2_DDR_OP_NAME) == std::string::npos) {
                found = true;
            }
        });
        if (found) {
            executeOps.push_back(execOp);
        }
    });

    if (executeOps.empty()) {  // No ExecuteOps with CopyOp in the network
        return;
    }

    // For each measured DMA operations two timestamps will be captured
    const unsigned elementSize = VPUIP::HW_DMA_PROFILING_SIZE_BYTES / sizeof(uint32_t);
    const unsigned output_size = static_cast<unsigned>(executeOps.size() * elementSize);

    // Calculate number of chunks and DMA operation inside one chunk
    // based on the maximum CMX buffer size
    const unsigned totalSizeBytes = output_size * sizeof(uint32_t);
    auto chunkWalker =
            vpux::ChunkWalker(totalSizeBytes, VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE, sizeof(uint32_t), _log);

    const auto cmxMemType =
            getMemRefType(ShapeRef({chunkWalker.getOpsInChunk()}), getUInt32Type(ctx), DimsOrder::C, memSpace);
    const auto cmxMemTypeLast =
            getMemRefType(ShapeRef({chunkWalker.getOpsInLastChunk()}), getUInt32Type(ctx), DimsOrder::C, memSpace);
    const auto outputResult = mlir::MemRefType::get({output_size}, getUInt32Type(ctx));

    // Declare and create additional output from network
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dma");

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    mlir::memref::AllocOp memOp;

    unsigned dmaId = 0;                      // Total DMA ops counter
    SmallVector<mlir::Value> timestampsOps;  // Collect chunk timestimps(Cleared inside AddCMX2DDRExecuteOp)
    SmallVector<mlir::Value> waitOps;        // Collect chunk results

    auto chunkSwitchCallback = [&](const unsigned chunkId, const unsigned opsInChunk, bool lastChunk) {
        if (chunkId) {
            waitOps.push_back(AddCMX2DDRExecuteOp(builder, ctx, profilingResult, memOp, timestampsOps, 1,
                                                  (chunkId - 1) * opsInChunk, "dma"));
        }
        builder.restoreInsertionPoint(lastInsertionPoint);
        memOp = builder.create<mlir::memref::AllocOp>(
                mlir::NameLoc::get(mlir::Identifier::get("dmaProfilingSubviewBuffer", ctx)),
                (!lastChunk) ? cmxMemType : cmxMemTypeLast);
        lastInsertionPoint = builder.saveInsertionPoint();
    };

    auto chunkItemCallback = [&](mlir::async::ExecuteOp execOp, const unsigned& chunkDmaId) {
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
                    SmallVector<int64_t>({static_cast<int64_t>(chunkDmaId)}), timestampType.getShape());
            std::string curTaskName;
            curTaskName = stringifyLocation(op->getLoc());
            auto name = mlir::NameLoc::get(
                    mlir::Identifier::get(curTaskName + ((!after) ? (dmaId == 0 ? "_PROFBEGIN" : "_PROFTASKBEGIN")
                                                                  : ("_PROFTASKEND_" + std::to_string(dmaId - 1) + "_" +
                                                                     std::to_string(dmaId / 2 + 1))),
                                          ctx));
            dmaId++;
            chunkWalker.increment();
            return builder.create<IERT::TimestampOp>(name, timestampType, sub).output();
        };
        SmallVector<mlir::Value> localTimestampsOps;
        localTimestampsOps.push_back(insertDma(firstCopy, false));
        localTimestampsOps.push_back(insertDma(lastCopy, true));

        // Prepare for execOp rebuilding: Add new results to the current yieldOp
        auto yieldOp = mlir::dyn_cast<mlir::async::YieldOp>(execOp.body().front().getTerminator());
        unsigned firstTimestampOperandId = static_cast<unsigned>(yieldOp.operands().size());
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

        auto executor = vpux::IERT::IERTDialect::getExecutor(execOp);
        IERT::IERTDialect::setExecutor(newExecOp, executor);

        for (size_t id = 0; id < localTimestampsOps.size(); id++) {
            timestampsOps.push_back(newExecOp.results()[firstTimestampOperandId + id]);
        }

        // Remove old execOp
        auto newResults = newExecOp->getResults().drop_back(localTimestampsOps.size());
        execOp->replaceAllUsesWith(newResults);
        execOp->erase();
    };
    chunkWalker.run<SmallVector<mlir::async::ExecuteOp>>(executeOps, chunkSwitchCallback, chunkItemCallback);

    // Copy to DDR the last chunk
    waitOps.push_back(AddCMX2DDRExecuteOp(builder, ctx, profilingResult, memOp, timestampsOps, 1,
                                          (chunkWalker.getChunks() - 1) * chunkWalker.getOpsInChunk(), "dma"));

    //
    // Concat all chunks together and push to the network returnOp
    //
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<IERT::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("dmaDDRProfiling", ctx)), waitOps, profilingResult);
    returnOp.operandsMutable().append(concatview.output());
}

// DPU profiling pass
// Add profiling buffer for the all DPU Clusters in the network
// Steps:
//   1. Allocate buffer in CMX for the first chunk(configured via HW_DPU_PROFILING_SIZE_BYTES)
//   2. Fill it with results of profiling data from DPU operations
//   3. Connect results to the ConcatOp
//   4. Send result of concatOp to DDR using new CopyOp
//   5. Allocate buffer for the next chunk and continue with steps 2-4
//   6. Connect all DMA to DDR operations to the ConcatOp and connect it to the new network profiling output
void DPUProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto memSpace = _memSpaceCb(ctx, "");
    if (memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<std::pair<VPUIP::NCEClusterTaskOp, int64_t>> dpuTasks;
    netFunc.walk([&](VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
        _log.trace("Process Operation '{0}'", nceClusterTaskOp->getLoc());

        auto dpuIt = nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>();
        auto count = std::distance(dpuIt.begin(), dpuIt.end());
        dpuTasks.push_back({nceClusterTaskOp, count});
    });

    if (dpuTasks.empty()) {  // No DPU task in the network
        return;
    }

    // [Track number: E#26531]
    // Enable profiling for all DPU tasks in the NCE cluster

    const unsigned elementSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES / sizeof(uint64_t);
    const unsigned output_size = static_cast<unsigned>(dpuTasks.size() * elementSize);

    // Calculate number of chunks and DMA operation inside one chunk
    // based on the maximum CMX buffer size
    const unsigned totalSizeBytes = output_size * sizeof(uint64_t);
    auto chunkWalker = vpux::ChunkWalker(totalSizeBytes, VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE,
                                         VPUIP::HW_DPU_PROFILING_SIZE_BYTES, _log);

    const auto cmxMemType = getMemRefType(ShapeRef({chunkWalker.getOpsInChunk() * elementSize}), getUInt64Type(ctx),
                                          DimsOrder::C, memSpace);
    const auto cmxMemTypeLast = getMemRefType(ShapeRef({chunkWalker.getOpsInLastChunk() * elementSize}),
                                              getUInt64Type(ctx), DimsOrder::C, memSpace);
    const auto outputResult = mlir::MemRefType::get({output_size}, getUInt64Type(ctx));

    // Declare and create additional output from network
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dpu");

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    mlir::memref::AllocOp memOp;

    unsigned dpuId = 0;  // Total DPU ops counter
    SmallVector<mlir::Value> dpuProfilingOutputs;
    SmallVector<mlir::Value> waitOps;  // Collect chunk results
    auto chunkSwitchCallback = [&](const unsigned chunkId, const unsigned opsInChunk, bool lastChunk) {
        if (chunkId) {
            waitOps.push_back(AddCMX2DDRCopyOp(builder, ctx, profilingResult, memOp, dpuProfilingOutputs, elementSize,
                                               (chunkId - 1) * opsInChunk, "dpu"));
        }
        builder.restoreInsertionPoint(lastInsertionPoint);
        memOp = builder.create<mlir::memref::AllocOp>(
                mlir::NameLoc::get(mlir::Identifier::get("dpuProfilingSubviewBuffer", ctx)),
                (!lastChunk) ? cmxMemType : cmxMemTypeLast);
        lastInsertionPoint = builder.saveInsertionPoint();
    };

    auto chunkItemCallback = [&](std::pair<VPUIP::NCEClusterTaskOp, int64_t> dpuTask, const unsigned& chunkDpuId) {
        auto cluster = dpuTask.first;
        builder.setInsertionPointAfter(cluster);
        auto timestampType = getMemRefType({elementSize}, getUInt64Type(ctx), DimsOrder::C, memSpace);
        auto sub = builder.create<IERT::SubViewOp>(
                mlir::NameLoc::get(mlir::Identifier::get("dpuProfilingSubview", ctx)), memOp,
                SmallVector<int64_t>({static_cast<int>(chunkDpuId * elementSize)}), timestampType.getShape());

        SmallVector<mlir::Type> newResultTypes(cluster.getResultTypes());
        newResultTypes.push_back(timestampType);
        const auto profilingMeta = llvm::formatv("_PROF_{0}", dpuId).str();
        const auto loc = appendLoc(cluster->getLoc(), profilingMeta);
        auto newCluster = builder.create<VPUIP::NCEClusterTaskOp>(loc, newResultTypes, cluster->getOperands(),
                                                                  cluster->getAttrs());

        for (const auto& region : llvm::enumerate(cluster.getRegions())) {
            newCluster.getRegion(static_cast<unsigned>(region.index())).takeBody(*region.value());
        }
        newCluster.profiling_dataMutable().assign(sub);
        dpuProfilingOutputs.push_back(newCluster.profiling_output());

        cluster->replaceAllUsesWith(mlir::ValueRange(newCluster.output()));
        cluster->erase();
        chunkWalker.increment();
        dpuId++;
    };
    chunkWalker.run<SmallVector<std::pair<VPUIP::NCEClusterTaskOp, int64_t>>>(dpuTasks, chunkSwitchCallback,
                                                                              chunkItemCallback);

    // Copy to DDR the last chunk
    waitOps.push_back(AddCMX2DDRCopyOp(builder, ctx, profilingResult, memOp, dpuProfilingOutputs, elementSize,
                                       (chunkWalker.getChunks() - 1) * chunkWalker.getOpsInChunk(), "dpu"));

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

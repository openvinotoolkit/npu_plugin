//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/IR/Attributes.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <numeric>
#include <sstream>

using namespace vpux;

namespace {

//
// DMATaskProfilingPass
//

class DMATaskProfilingPass final : public VPUIP::DMATaskProfilingBase<DMATaskProfilingPass> {
public:
    explicit DMATaskProfilingPass(VPUIP::MemKindCreateFunc memKindCb, Logger log): _memKindCb(std::move(memKindCb)) {
        VPUX_THROW_UNLESS(_memKindCb != nullptr, "Missing memKindCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    VPUIP::MemKindCreateFunc _memKindCb;
};

//
// ActShaveProfilingPass
//

class ActShaveProfilingPass final : public VPUIP::ActShaveProfilingBase<ActShaveProfilingPass> {
public:
    explicit ActShaveProfilingPass(VPUIP::MemKindCreateFunc memKindCb, Logger log): _memKindCb(std::move(memKindCb)) {
        VPUX_THROW_UNLESS(_memKindCb != nullptr, "Missing memKindCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPU::MemoryKind _memKind{vpux::VPU::MemoryKind::DDR};
};

mlir::Value AddCMX2DDRExecuteOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::BlockArgument& profilingResult,
                                mlir::Value cmxMemOp, SmallVector<mlir::Value>& timestampsOps, unsigned elementSize,
                                unsigned offset, StringRef name) {
    auto elementType = cmxMemOp.getType().cast<mlir::MemRefType>().getElementType();
    const auto resultType =
            mlir::MemRefType::get({static_cast<int64_t>(timestampsOps.size() * elementSize)}, elementType);

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
    auto sub = builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "DDR" + std::to_string(offset), ctx)), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * elementSize}), resultType.getShape());
    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "Profiling" + std::to_string(offset), ctx)), values,
            cmxMemOp);
    auto outputOp = builder.create<VPUIP::CopyOp>(copyLoc, concatview.output(), sub);
    builder.create<mlir::async::YieldOp>(copyLoc, outputOp->getResults());

    // Add execution attributes to async exec op
    auto newOpExecutor = mlir::dyn_cast_or_null<VPUIP::AsyncLayerOpInterface>(outputOp.getOperation());
    auto executor = newOpExecutor.getExecutor();
    if (executor != nullptr) {
        VPUIP::VPUIPDialect::setExecutor(execOp, executor);
    }
    builder.setInsertionPointAfter(execOp);
    auto waitOp = builder.create<mlir::async::AwaitOp>(execOp->getLoc(), execOp.results()[0]);

    timestampsOps.clear();
    return waitOp.result();
};

mlir::Value AddCMX2DDRCopyOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::BlockArgument& profilingResult,
                             mlir::Value cmxMemOp, SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned elementSize,
                             unsigned numElements, unsigned offset, StringRef name) {
    auto elementType = cmxMemOp.getType().cast<mlir::MemRefType>().getElementType();
    const auto resultType = mlir::MemRefType::get({static_cast<int64_t>(numElements * elementSize)}, elementType);

    auto subDDR = builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "DDR" + std::to_string(offset), ctx)), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * elementSize}), resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc2 =
            mlir::NameLoc::get(mlir::Identifier::get(name + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset), ctx));
    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get(name + "ProfilingConcat" + std::to_string(offset), ctx)),
            dpuProfilingOutputs, cmxMemOp);

    dpuProfilingOutputs.clear();
    return builder.create<VPUIP::CopyOp>(copyLoc2, concatview.output(), subDDR).output();
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

    auto maybeMemKind = _memKindCb("");
    if (!maybeMemKind.hasValue()) {
        _log.trace("Memory Space is not defined");
        return;
    }

    vpux::IndexedSymbolAttr memKindAttr = nullptr;
    {
        const auto memKind = maybeMemKind.getValue();
        if (memKind == VPU::MemoryKind::CMX_NN) {
            memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), 0);
        } else {
            memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(memKind));
        }
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<mlir::async::ExecuteOp> executeOps;
    mlir::MemRefType timestampType;
    const auto arch = VPU::getArch(module);
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        timestampType = getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, memKindAttr);
        break;
    case VPU::ArchKind::VPUX37XX:
        timestampType = getMemRefType(ShapeRef({1}), getUInt64Type(ctx), DimsOrder::C, memKindAttr);
        break;
    default:
        VPUX_THROW("Not supported architecture");
    }

    // Find all execOp which contains CopyOps
    netFunc.walk([&](mlir::async::ExecuteOp execOp) {
        _log.trace("Process Operation '{0}'", execOp->getLoc());

        bool found = false;
        auto& bodyBlock = execOp.body().front();
        bodyBlock.walk([&](VPUIP::CopyOp curTask) {
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

    const auto cmxMemType = getMemRefType(ShapeRef({chunkWalker.getOpsInChunk()}), timestampType.getElementType(),
                                          DimsOrder::C, memKindAttr);
    const auto cmxMemTypeLast = getMemRefType(ShapeRef({chunkWalker.getOpsInLastChunk()}),
                                              timestampType.getElementType(), DimsOrder::C, memKindAttr);
    const auto outputResult = mlir::MemRefType::get({output_size}, timestampType.getElementType());

    // Declare and create additional output from network
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dma");

    builder.setInsertionPoint(&netFunc.getBody().front().front());
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
        bodyBlock.walk([&](VPUIP::CopyOp curTask) {
            lastCopy = curTask.getOperation();
            if (firstCopy == nullptr)
                firstCopy = lastCopy;
        });

        //
        // Insertion of Timestamp Ops to the current execOp
        //
        auto insertDma = [&](mlir::Operation* op, bool after) {
            auto* insertionPoint = op;
            VPUIP::NCEClusterTilingOp nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op->getParentOp());
            // In case CopyOp is wrapped with NCEClusterTiling then new TimestampOps
            // should be added around NCEClusterTiling
            if (nceClusterTilingOp) {
                insertionPoint = nceClusterTilingOp.getOperation();
            }

            if (after) {
                builder.setInsertionPointAfter(insertionPoint);
            } else {
                builder.setInsertionPoint(insertionPoint);
            }
            auto sub = builder.create<VPUIP::SubViewOp>(
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
            return builder.create<VPUIP::TimestampOp>(name, timestampType, sub).output();
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

        auto executor = vpux::VPUIP::VPUIPDialect::getExecutor(execOp);
        VPUIP::VPUIPDialect::setExecutor(newExecOp, executor);

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

    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("dmaDDRProfiling", ctx)), waitOps, profilingResult);
    returnOp.operandsMutable().append(concatview.output());
}

void ActShaveProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto maybeMemKind = _memKindCb("");
    if (!maybeMemKind.hasValue()) {
        _log.trace("Memory Space is not defined");
        return;
    }

    vpux::IndexedSymbolAttr memKindAttr = nullptr;
    {
        const auto memKind = maybeMemKind.getValue();
        if (memKind == VPU::MemoryKind::CMX_NN) {
            memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), 0);
        } else {
            memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(memKind));
        }
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<VPUIP::SwKernelOp> swTasks;
    bool isClusterTilingAppliedToActShaves = false;
    netFunc.walk([&](VPUIP::SwKernelOp swKernelOp) {
        _log.trace("Process Operation '{0}'", swKernelOp->getLoc());

        isClusterTilingAppliedToActShaves |=
                (mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp()) != nullptr);
        swTasks.push_back(swKernelOp);
    });

    // Either no ActShave tasks in the network or cluster tiling applied (which is not supported yet)
    if (swTasks.empty() || isClusterTilingAppliedToActShaves) {
        return;
    }

    const unsigned elementSize = VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES / sizeof(uint32_t);
    const unsigned output_size = static_cast<unsigned>(swTasks.size() * elementSize);
    auto profilingType = getMemRefType({elementSize}, getUInt32Type(ctx), DimsOrder::C, memKindAttr);

    // Calculate number of chunks and operations inside one chunk
    // based on the maximum CMX buffer size
    const unsigned totalSizeBytes = output_size * sizeof(uint32_t);
    auto chunkWalker = vpux::ChunkWalker(totalSizeBytes, VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE,
                                         VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES, _log);

    const auto cmxMemType = getMemRefType(ShapeRef({chunkWalker.getOpsInChunk() * elementSize}),
                                          profilingType.getElementType(), DimsOrder::C, memKindAttr);
    const auto cmxMemTypeLast = getMemRefType(ShapeRef({chunkWalker.getOpsInLastChunk() * elementSize}),
                                              profilingType.getElementType(), DimsOrder::C, memKindAttr);
    const auto outputResult = mlir::MemRefType::get({output_size}, profilingType.getElementType());

    // Declare and create additional output from network
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "actshave");

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    mlir::memref::AllocOp memOp;

    unsigned taskId = 0;
    SmallVector<mlir::Value> shaveProfilingOutputs;
    SmallVector<mlir::Value> waitOps;  // Collect chunk results
    auto chunkSwitchCallback = [&](const unsigned chunkId, const unsigned opsInChunk, bool lastChunk) {
        if (chunkId) {
            waitOps.push_back(AddCMX2DDRCopyOp(builder, ctx, profilingResult, memOp, shaveProfilingOutputs, elementSize,
                                               opsInChunk, (chunkId - 1) * opsInChunk, "actshave"));
        }
        builder.restoreInsertionPoint(lastInsertionPoint);
        memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx),
                                                      (!lastChunk) ? cmxMemType : cmxMemTypeLast);
        lastInsertionPoint = builder.saveInsertionPoint();
    };

    auto chunkItemCallback = [&](VPUIP::SwKernelOp shaveTask, const unsigned& chunkDpuId) {
        builder.setInsertionPointAfter(shaveTask);
        auto sub = builder.create<VPUIP::SubViewOp>(
                mlir::NameLoc::get(mlir::Identifier::get("actshaveProfilingSubview", ctx)), memOp,
                SmallVector<int64_t>({static_cast<int>(chunkDpuId * elementSize)}), profilingType.getShape());

        SmallVector<mlir::Type> newResultTypes(shaveTask.getResultTypes());
        newResultTypes.push_back(profilingType);
        const auto profilingMeta = llvm::formatv("_PROF_{0}", taskId).str();
        const auto loc = appendLoc(shaveTask->getLoc(), profilingMeta);
        auto newShaveTask =
                builder.create<VPUIP::SwKernelOp>(loc, shaveTask.inputs(), shaveTask.output_buffs(), sub.result(),
                                                  shaveTask.kernelFunction(), shaveTask.tileIndexAttr());

        newShaveTask.getRegion().takeBody(shaveTask.getRegion());
        shaveProfilingOutputs.push_back(newShaveTask.profiling_output());

        shaveTask->replaceAllUsesWith(newShaveTask.results());
        shaveTask->erase();
        chunkWalker.increment();
        taskId++;
    };
    chunkWalker.run<SmallVector<VPUIP::SwKernelOp>>(swTasks, chunkSwitchCallback, chunkItemCallback);

    // Copy to DDR the last chunk
    waitOps.push_back(AddCMX2DDRCopyOp(builder, ctx, profilingResult, memOp, shaveProfilingOutputs, elementSize,
                                       chunkWalker.getOpsInLastChunk(),
                                       (chunkWalker.getChunks() - 1) * chunkWalker.getOpsInChunk(), "actshave"));

    //
    // Concat all chunks together and push to the network returnOp
    //
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::Identifier::get("actshaveDDRProfiling", ctx)), waitOps, profilingResult);
    returnOp.operandsMutable().append(concatview.output());
}

//
// UPAProfilingPass
//

class UPAProfilingPass final : public VPUIP::UPAProfilingBase<UPAProfilingPass> {
public:
    explicit UPAProfilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// GroupProfilingBuffersPass
//

class GroupProfilingBuffersPass final : public VPUIP::GroupProfilingBuffersBase<GroupProfilingBuffersPass> {
public:
    explicit GroupProfilingBuffersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void UPAProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<VPURT::TaskOp> upaTasks;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() == vpux::VPU::ExecutorKind::SHAVE_UPA) {
            _log.trace("Adding Operation '{0}'", taskOp->getLoc());
            upaTasks.push_back(taskOp);
        }
    });

    if (upaTasks.empty()) {  // No UPA task in the network
        return;
    }

    unsigned elementSize = VPUIP::HW_UPA_PROFILING_SIZE_BYTES / sizeof(uint32_t);
    unsigned outputSize = static_cast<unsigned>(upaTasks.size() * elementSize);
    auto outputResult = mlir::MemRefType::get({outputSize}, getUInt32Type(ctx));

    auto profilingId = static_cast<int64_t>(netOp.getProfilingOutputsCount());
    unsigned upaId = 0;
    for (auto& upaTask : upaTasks) {
        builder.setInsertionPoint(upaTask);
        auto timestampType = mlir::MemRefType::get({elementSize}, getUInt32Type(ctx));
        int offset = upaId * elementSize * sizeof(uint32_t);
        auto declareOp = builder.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::Identifier::get("declareProfilingBuffer", ctx)), timestampType,
                VPURT::BufferSection::ProfilingOutput, profilingId, offset);

        const auto loc = appendLoc(upaTask->getLoc(), "_PROF_{0}", upaId);
        upaTask->setLoc(loc);
        upaTask.profiling_dataMutable().assign(declareOp);
        upaId++;
    }

    // Declare and create additional output from network
    auto profilngResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "upa");

    // And to the returnOp
    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    returnOp.operandsMutable().append(profilngResult);
}

void GroupProfilingBuffersPass::safeRunOnModule() {
    auto ctx = &getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    // If profiling was enabled for SW kernels only and network doesn't have any, there will be
    // profilingOutputsInfo with single block "^bb0" and 0 operation inside.
    if (netOp.profilingOutputsInfo().empty() || netOp.profilingOutputsInfo().front().empty() ||
        netOp.profilingOutputsInfo().front().getOps().empty()) {
        return;
    }

    // Collecting sizes of all profiling buffers in order to calculate offsets in the base buffer
    // New buffer name will be like: [offset]_[name]_[offset]_[name]...
    auto& profilingOutputs = netOp.profilingOutputsInfo().front();
    SmallVector<uint32_t> outputBases;
    uint32_t totalSize = 0;
    std::string newOutputName;
    profilingOutputs.walk([&](IE::DataInfoOp op) {
        outputBases.push_back(totalSize);
        auto type = op.userType().cast<mlir::RankedTensorType>();
        newOutputName += std::to_string(totalSize) + '_' + op.name().str() + '_';
        auto size = static_cast<uint32_t>(type.getSizeInBits() / CHAR_BIT);
        totalSize += size;
        op.erase();
    });
    newOutputName.pop_back();

    //
    // Create and add new combined profiling output to the user info
    //
    auto newOutputResult =
            mlir::MemRefType::get({static_cast<int64_t>(totalSize / sizeof(uint32_t))}, getUInt32Type(ctx));
    auto newOutputShapedType = newOutputResult.cast<vpux::NDTypeInterface>();
    auto outputUserResult = getTensorType(newOutputShapedType.getShape(), newOutputShapedType.getElementType(),
                                          newOutputShapedType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&profilingOutputs.front(), &builderLog);
    userInfoBuilder.create<IE::DataInfoOp>(
            mlir::NameLoc::get(mlir::Identifier::get("combinedProfilingDataOutputInfo", ctx)),
            mlir::StringAttr::get(ctx, newOutputName), mlir::TypeAttr::get(outputUserResult));

    auto totalArgumentsCount = netFunc.getNumArguments();
    auto mainArgumentsCount = totalArgumentsCount - outputBases.size();
    VPUX_THROW_UNLESS(mainArgumentsCount > 0, "There is no main network arguments in the funcOp");

    // Adding new output buffer
    auto newProfilngResult = netFunc.getBody().front().addArgument(newOutputResult);
    SmallVector<unsigned> argsToErase;
    unsigned removedArgs = 0;
    // Loop thought the old function arguments in order to replace its usages
    // with DeclareBufferOp with previously calculated offset connected to the new buffer
    // and erase these arguments
    auto arg = netFunc.getArguments().begin();
    while (arg != netFunc.getArguments().end()) {
        auto argNum = arg->getArgNumber();
        if (argNum > mainArgumentsCount - 1) {
            while (!arg->use_empty()) {
                auto use = arg->use_begin();
                auto op = use->getOwner();
                auto taskOp = op->getParentOfType<VPURT::TaskOp>();
                builder.setInsertionPoint((taskOp != nullptr) ? taskOp : op);
                unsigned base = (removedArgs) ? outputBases[removedArgs] : 0;
                auto declareOp = builder.create<VPURT::DeclareBufferOp>(
                        mlir::NameLoc::get(mlir::Identifier::get("newProfilingBuffer", ctx)), arg->getType(),
                        VPURT::BufferSection::ProfilingOutput, 0, base);
                use->set(declareOp.buffer());
            }
            netFunc.eraseArgument(argNum);
            removedArgs++;
            // Reach the last old profiling output, stop here as the next output is the new buffer
            if (removedArgs == outputBases.size()) {
                break;
            }
        } else {
            arg++;
        }
    }

    //
    // Recalculate existing DeclareTensorOp
    //
    netFunc.walk([&](VPURT::DeclareBufferOp op) {
        if (op.section() == VPURT::BufferSection::ProfilingOutput) {
            auto sectionIndex = op.sectionIndex();
            if (sectionIndex.hasValue()) {
                VPUX_THROW_UNLESS(sectionIndex.getValue().size() == 1,
                                  "Profiling output is expected to have just one locale index");
                auto idx = parseIntArrayAttr<int64_t>(sectionIndex.getValue())[0];
                if (idx <= 0) {
                    return;
                }

                auto base = outputBases[idx];
                op.sectionIndexAttr(getIntArrayAttr(ctx, SmallVector<int64_t>({0})));
                auto offset = static_cast<uint32_t>(base + op.byteOffset());
                op.byteOffsetAttr(builder.getUI32IntegerAttr(offset));
            }
        }
    });

    //
    // Replace function signature
    //
    auto funcType = netFunc.getType();
    auto newResultTypes = to_small_vector(llvm::concat<const mlir::Type>(
            funcType.getResults().drop_back(outputBases.size()), makeArrayRef(newOutputResult)));
    auto newInputsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(newOutputResult)));

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
    netFunc.setType(newFunctionType);

    //
    // Replace function return operands
    //
    netFunc.walk([&](mlir::ReturnOp op) {
        auto start = static_cast<unsigned>(op.operandsMutable().size() - outputBases.size());
        op.operandsMutable().erase(start, static_cast<unsigned>(outputBases.size()));
        op.operandsMutable().append(newProfilngResult);
    });
}

}  // namespace

//
// createDMATaskProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMATaskProfilingPass(MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<DMATaskProfilingPass>(std::move(memKindCb), log);
}

//
// createActShaveProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createActShaveProfilingPass(MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<ActShaveProfilingPass>(std::move(memKindCb), log);
}

//
// createUPAProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUPAProfilingPass(Logger log) {
    return std::make_unique<UPAProfilingPass>(log);
}

//
// createGroupProfilingBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createGroupProfilingBuffersPass(Logger log) {
    return std::make_unique<GroupProfilingBuffersPass>(log);
}

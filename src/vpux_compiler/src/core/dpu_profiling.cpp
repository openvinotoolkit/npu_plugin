//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/dpu_profiling.hpp"

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <numeric>
#include <sstream>

namespace vpux {

using namespace vpux;

// Return number of used clusters
unsigned getClustersNumber(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    std::set<uint64_t> clusterIds;
    for (auto dpuTask : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
        const auto clusterId = dpuTask.cluster_id().getValueOr(0);
        clusterIds.insert(clusterId);
    }
    return static_cast<unsigned>(clusterIds.size());
}

template <class T>
unsigned countDpuTasks(SmallVector<std::pair<T, unsigned>> vector) {
    return std::accumulate(vector.begin(), vector.end(), 0, [](const auto& a, const auto& b) {
        return a + b.second;
    });
}

mlir::Type BaseClusterBufferScheduler::getTimestampType(unsigned dpuTasksAmount) {
    return getMemRefType({_profilingElementSize * dpuTasksAmount}, getUInt64Type(_ctx), DimsOrder::C, _memKindAttr);
}

BaseClusterBufferScheduler::BaseClusterBufferScheduler(unsigned clustersAmount, unsigned profilingWorkloadSize,
                                                       mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                                       vpux::VPU::MemoryKind memKind, mlir::FuncOp netFunc,
                                                       std::shared_ptr<NameUniqifier> uniqifier)
        : _clustersAmount(clustersAmount),
          _profilingWorkloadSize(profilingWorkloadSize),
          _profilingElementSize(profilingWorkloadSize /
                                sizeof(uint64_t)),  // How many words are need to store one workload
          _profilingBufferSizes({0}),
          _builder(builder),
          _ctx(ctx),
          _netFunc(netFunc),
          _memKindAttr(IndexedSymbolAttr::get(ctx, stringifyEnum(memKind))),
          _uniqifier(uniqifier) {
}

unsigned BaseClusterBufferScheduler::getRequiredDdrMemory() const {
    unsigned dpuTasksAmount =
            std::accumulate(_nceTaskSignatures.begin(), _nceTaskSignatures.end(), 0, [](const auto& a, const auto& b) {
                return a + b._maxSubTasks;
            });
    return dpuTasksAmount * _clustersAmount * _profilingElementSize;
}

void BaseClusterBufferScheduler::scheduleNceTask(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    const auto taskSignature = getTaskSignature(nceClusterTaskOp);
    const auto maxDpuTasks = taskSignature._maxSubTasks;

    const auto requiredMemory = maxDpuTasks * _profilingWorkloadSize;
    VPUX_THROW_WHEN(requiredMemory > VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE,
                    "NCEClusterTask at '{0}' requires more memory {1} then currently supported. Change  "
                    "HW_DPU_PROFILING_MAX_BUFFER_SIZE.",
                    nceClusterTaskOp->getLoc(), requiredMemory);
    _nceTaskSignatures.push_back(taskSignature);
    // Trying to reuse last profiling buffer
    const auto currentBufferSize = _profilingBufferSizes.back();
    const auto newBufferSize = currentBufferSize + maxDpuTasks;
    // If we can store profiling result of current task in last buffer without exceeding
    // max size - reuse it, otherwise - scheduling one more
    if (newBufferSize * _profilingWorkloadSize > VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE) {
        _profilingBufferSizes.push_back(maxDpuTasks);
    } else {
        _profilingBufferSizes.pop_back();
        _profilingBufferSizes.push_back(newBufferSize);
    }
}

void BaseClusterBufferScheduler::addProfilingOps(unsigned& currentDDROffset, SmallVector<mlir::Value>& clusterResults,
                                                 mlir::BlockArgument& profilingResult, int& nceId) {
    if (getRequiredDdrMemory() == 0) {
        return;
    }
    // Contains profiling_output of individual nceTaskOp and amount of profiled DPU tasks
    SmallVector<std::pair<mlir::Value, unsigned>> nceProfilingOutputs;
    mlir::Operation* currentProfilingBuffer = nullptr;
    int currentBufferId = -1;
    const auto allocateProfilingBufferCMX = [&]() {
        if (_profilingBufferSizes.empty()) {
            return;
        }

        ++currentBufferId;
        const auto currentBufferSize = _profilingBufferSizes.front();
        VPUX_THROW_WHEN(currentBufferSize == 0, "Empty CMXBuffers is not allowed");

        _profilingBufferSizes.pop_front();

        const unsigned totalSizeCMXElements = currentBufferSize * _profilingElementSize * _clustersAmount;
        const auto locationName =
                std::to_string(_clustersAmount) + "_dpuProfilingSubviewBuffer_" + std::to_string(currentBufferId);

        mlir::OpBuilder::InsertPoint lastInsertionPoint = _builder.saveInsertionPoint();
        _builder.setInsertionPointAfter(&_netFunc.getBody().front().front());

        currentProfilingBuffer = createAllocationOp(totalSizeCMXElements, locationName);

        _builder.restoreInsertionPoint(lastInsertionPoint);
    };

    const auto flushCMX2DDR = [&]() {
        if (nceProfilingOutputs.empty() || currentProfilingBuffer == nullptr) {
            return;
        }

        const auto flushedTasksAmount = countDpuTasks(nceProfilingOutputs);
        SmallVector<mlir::Value> profilingOutputs;
        std::transform(nceProfilingOutputs.begin(), nceProfilingOutputs.end(), std::back_inserter(profilingOutputs),
                       [](const auto& x) {
                           return x.first;
                       });

        clusterResults.push_back(copyToDDR(profilingResult, currentProfilingBuffer, profilingOutputs,
                                           flushedTasksAmount, currentDDROffset, "dpu"));

        profilingOutputs.clear();
        nceProfilingOutputs.clear();
        currentDDROffset += flushedTasksAmount;
    };

    // Allocate first buffer for storing profiling results
    allocateProfilingBufferCMX();
    for (auto& nceTaskSignature : _nceTaskSignatures) {
        auto nceTaskOp = nceTaskSignature._task;
        auto* insertionPoint = nceTaskOp.getOperation();
        auto nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(nceTaskOp->getParentOp());
        // In case NCE task is wrapped with NCEClusterTiling then inserting should be done
        // at NCEClusterTiling op level and not inside it where NCEClusterTask op is
        if (nceClusterTilingOp) {
            insertionPoint = nceClusterTilingOp.getOperation();
        }
        _builder.setInsertionPoint(insertionPoint);

        const unsigned dpuTasksAmount = nceTaskSignature._maxSubTasks * _clustersAmount;
        auto profilingSamplesInCMX = countDpuTasks(nceProfilingOutputs);
        const auto expectedCMXMemoryUsage = (profilingSamplesInCMX + dpuTasksAmount) * _profilingWorkloadSize;
        // If couldnt place current task in the end of cmx buffer flushing all previous tasks to DDR
        // expectedCMXMemoryUsage counts size for all clusters, while HW_DPU_PROFILING_MAX_BUFFER_SIZE only for one
        // so, need to align them for comparison
        if (expectedCMXMemoryUsage > VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE * _clustersAmount) {
            flushCMX2DDR();  // Flush current CMX content to DDR
            profilingSamplesInCMX = 0;
            allocateProfilingBufferCMX();  // Allocate next CMX buffer
        }

        const SmallVector<int64_t> sizes(
                {static_cast<int64_t>(dpuTasksAmount) * static_cast<int64_t>(_profilingElementSize)});
        auto subView = getViewToBuffer(currentProfilingBuffer, profilingSamplesInCMX, sizes);
        auto subViewType = subView.getType();

        auto timestampType = getTimestampType(dpuTasksAmount);
        if (auto distrType = subViewType.dyn_cast<VPUIP::DistributedBufferType>()) {
            timestampType = distrType.getCompactType();
        }

        const auto profilingMeta = nceTaskSignature.signature(nceId, currentBufferId);
        const auto loc = appendLoc(_uniqifier->getUniqueLoc(nceTaskOp->getLoc()), profilingMeta);

        _builder.setInsertionPointAfter(nceTaskOp);

        const auto outputType = nceTaskOp.output().getType();
        const auto outputSMType = nceTaskOp.output_sparsity_map() ? nceTaskOp.output_sparsity_map().getType() : nullptr;
        auto newCluster = _builder.create<VPUIP::NCEClusterTaskOp>(loc, outputType, outputSMType, timestampType,
                                                                   nceTaskOp->getOperands(), nceTaskOp->getAttrs());
        for (const auto& region : llvm::enumerate(nceTaskOp.getRegions())) {
            newCluster.getRegion(static_cast<unsigned>(region.index())).takeBody(*region.value());
        }
        newCluster.profiling_dataMutable().assign(subView);
        SmallVector<mlir::Value> newUses{newCluster.output()};
        if (newCluster.output_sparsity_map() != nullptr) {
            newUses.push_back(newCluster.output_sparsity_map());
        }
        nceTaskOp->replaceAllUsesWith(mlir::ValueRange(newUses));
        nceTaskOp->erase();
        auto profilingOutput = newCluster.profiling_output();

        // In case original NCEClusterTask was wrapped with NCEClusterTiling then new NCEClusterTask
        // with additional profiling output should also be wrapped with NCEClusterTiling op whose
        // list of operands and results were extended for profiling buffer
        if (nceClusterTilingOp) {
            _builder.setInsertionPoint(insertionPoint);

            // Operands of new NCEClusterTilingOp will be extended with profiling buffer
            SmallVector<mlir::Value> newNceClusterTilingOperands(nceClusterTilingOp->getOperands());
            newNceClusterTilingOperands.push_back(newCluster.profiling_data());

            // Result of new NCEClusterTilingOp will be extended with profiling result
            SmallVector<mlir::Type> newNceClusterTilingResultTypes(nceClusterTilingOp->getResultTypes());
            newNceClusterTilingResultTypes.push_back(subViewType);

            const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
                std::ignore = loc;

                mlir::BlockAndValueMapping mapper;

                auto origArguments = nceClusterTilingOp.body().front().getArguments();

                // Map original NCEClusterTiling argument to new corresponding operands and map
                // profiling buffer to last operand
                mapper.map(origArguments, newOperands.take_front(nceClusterTilingOp->getOperands().size()));
                mapper.map(newCluster.profiling_data(), newOperands.take_back(1).front());

                builder.clone(*newCluster.getOperation(), mapper);
            };

            auto newNceClusterTilingOp = _builder.create<VPUIP::NCEClusterTilingOp>(
                    nceClusterTilingOp->getLoc(), newNceClusterTilingResultTypes, newNceClusterTilingOperands,
                    bodyBuilder);

            // Replace all uses of old NCEClusterTiling op with new one
            // except newly added profiling output
            auto newResults = newNceClusterTilingOp->getResults().drop_back(1);
            nceClusterTilingOp->replaceAllUsesWith(newResults);

            // Remove old NCEClusterTiling inner task and task itself
            nceClusterTilingOp.getInnerTaskOp()->erase();
            nceClusterTilingOp->erase();

            // Set new insertion point back at new NCEClusterTiling level
            insertionPoint = newNceClusterTilingOp.getOperation();
            _builder.setInsertionPointAfter(insertionPoint);

            // Store information about profiling result which later is concatenated with rest of profiling data
            // and copied from buffer in CMX to DDR
            profilingOutput = newNceClusterTilingOp.getResult(newNceClusterTilingOp->getNumResults() - 1);
        }
        nceProfilingOutputs.push_back({profilingOutput, dpuTasksAmount});
        nceId++;
    }
    flushCMX2DDR();
}

SingleClusterScheduler::SingleClusterScheduler(unsigned profilingWorkloadSize, mlir::OpBuilder& builder,
                                               mlir::MLIRContext* ctx, vpux::VPU::MemoryKind memKind,
                                               mlir::FuncOp netFunc, std::shared_ptr<NameUniqifier> uniqifier)
        : BaseClusterBufferScheduler(1, profilingWorkloadSize, builder, ctx, memKind, netFunc, uniqifier) {
    if (memKind == VPU::MemoryKind::CMX_NN) {
        _memKindAttr = IndexedSymbolAttr::get(_ctx, stringifyEnum(memKind), 0);
    } else {
        _memKindAttr = IndexedSymbolAttr::get(_ctx, stringifyEnum(memKind));
    }
}

NCETaskSignature SingleClusterScheduler::getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    const auto dpuIt = nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>();
    const auto maxDpuTasks = static_cast<unsigned>(std::distance(dpuIt.begin(), dpuIt.end()));
    return {nceClusterTaskOp, maxDpuTasks, {maxDpuTasks}};
}

mlir::Operation* SingleClusterScheduler::createAllocationOp(unsigned totalSizeCMXElements,
                                                            const std::string& location) {
    const auto cmxMemType =
            getMemRefType(ShapeRef(totalSizeCMXElements), getUInt64Type(_ctx), DimsOrder::C, _memKindAttr);

    return _builder.create<mlir::memref::AllocOp>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                  cmxMemType);
}

mlir::Value SingleClusterScheduler::copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                              SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements,
                                              unsigned offset, StringRef name) {
    const auto resultType = mlir::MemRefType::get(
            {static_cast<int64_t>(numElements) * static_cast<int64_t>(_profilingElementSize)}, getUInt64Type(_ctx));

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "DDR" + std::to_string(offset))), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * _profilingElementSize}), resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc2 = mlir::NameLoc::get(
            mlir::StringAttr::get(_ctx, name + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "ProfilingConcat" + std::to_string(offset))),
            dpuProfilingOutputs, cmxMemOp->getResult(0));

    return _builder.create<VPUIP::CopyOp>(copyLoc2, concatview.output(), subDDR).output();
}

mlir::Value SingleClusterScheduler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                    unsigned profilingSamplesInCMX, SmallVector<int64_t> sizes) {
    return _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "dpuProfilingSubview")),
            currentProfilingBuffer->getResult(0),
            SmallVector<int64_t>({static_cast<int>(profilingSamplesInCMX * _profilingElementSize)}), sizes);
}

VPUIP::DistributedBufferType MultiClusterScheduler::getDistributedBufferType(unsigned totalElements) {
    const auto layout = mlir::AffineMapAttr::get(DimsOrder::C.toAffineMap(_ctx));

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(_ctx, VPU::DistributionMode::SEGMENTED);
    const SmallVector<uint64_t> tiles = {_clustersAmount};
    const auto numTiles = getIntArrayAttr(_ctx, tiles);
    const auto numClusters = getIntAttr(_ctx, _clustersAmount);
    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTiles, nullptr, nullptr,
                                                                 nullptr, numClusters, nullptr, _ctx);
    return VPUIP::DistributedBufferType::get(_ctx, {totalElements}, getUInt64Type(_ctx), layout, _memKindAttr,
                                             distributedTensorAttr);
}

mlir::Type MultiClusterScheduler::getDistributedTimestampType(unsigned dpuTasksAmount) {
    return getDistributedBufferType(dpuTasksAmount * _profilingElementSize);
}

NCETaskSignature MultiClusterScheduler::getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
    SmallVector<unsigned> dpuTasksPerCluster(_clustersAmount, 0);
    unsigned maxTasksInCluster = 0;
    for (auto dpuTask : nceClusterTaskOp.variants().getOps<VPUIP::DPUTaskOp>()) {
        const auto clusterId = dpuTask.cluster_id().getValue();
        maxTasksInCluster = std::max(maxTasksInCluster, ++dpuTasksPerCluster[clusterId]);
    }
    return {nceClusterTaskOp, maxTasksInCluster, dpuTasksPerCluster};
}

mlir::Operation* MultiClusterScheduler::createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) {
    const auto bufferType = getDistributedBufferType(totalSizeCMXElements);
    return _builder.create<VPURT::AllocDistributed>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                    bufferType, nullptr, nullptr);
}

mlir::Value MultiClusterScheduler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                   unsigned profilingSamplesInCMX, SmallVector<int64_t> sizes) {
    return _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "dpuProfilingSubview")),
            currentProfilingBuffer->getResult(0),
            SmallVector<int64_t>({static_cast<int>(profilingSamplesInCMX * _profilingElementSize / _clustersAmount)}),
            sizes);
}

mlir::Value MultiClusterScheduler::copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation* cmxMemOp,
                                             SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements,
                                             unsigned offset, StringRef name) {
    const auto memorySize = numElements * _profilingElementSize;
    const auto resultTypeDDR = mlir::MemRefType::get({static_cast<int64_t>(memorySize)}, getUInt64Type(_ctx));
    const auto resultTypeDistributed = getDistributedBufferType(memorySize);

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "DDR" + std::to_string(offset))), profilingResult,
            SmallVector<int64_t>({static_cast<int64_t>(offset) * _profilingElementSize}), resultTypeDDR.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(
            mlir::StringAttr::get(_ctx, name + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(offset)));
    const auto concatLoc =
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, name + "ProfilingConcat" + std::to_string(offset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(concatLoc, resultTypeDistributed, dpuProfilingOutputs,
                                                           cmxMemOp->getResult(0));

    SmallVector<mlir::Value> inputsOutputOperands{concatview.output(), subDDR.result()};

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };

    return _builder.create<VPUIP::NCEClusterTilingOp>(copyLoc, resultTypeDDR, inputsOutputOperands, bodyBuilder)
            .getResult(0);
}

}  // namespace vpux

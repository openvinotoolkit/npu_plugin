//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include "vpux/compiler/core/act_profiling.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <deque>
#include <iterator>
#include <sstream>
#include <string>

namespace vpux {

using namespace vpux;

// Create a string that should be placed as a suffix for operation name (Loc) with relevant metadata
// allowing post processing tools to correctly interpret profiling data
std::string createActShaveProfilingLocSuffix(size_t inDdrOffset, size_t clusterSize, size_t inClusterOffset,
                                             Optional<size_t> tileId) {
    return formatv("_PROF_{0}_{1}_{2}_{3}", inDdrOffset, clusterSize, inClusterOffset, tileId).str();
}

// Gather profiling metadata from a profiled ActShave task
// Returns tuple: (size_t inDdrOffset, size_t clusterSize, size_t inClusterOffset and optional tile id)
std::tuple<size_t, size_t, size_t, Optional<size_t>> parseActShaveProfilingOffsets(mlir::Location loc) {
    const auto PROFILING_PREFIX_SIZE = std::string("_PROF_").size();
    const auto NUM_MANDATORY_FIELDS = 3;
    std::vector<size_t> values;
    Optional<size_t> maybeTileId = {};
    if (auto profLoc = loc.dyn_cast<mlir::FusedLoc>()) {
        const auto strProfLoc = stringifyLocation(profLoc.getLocations().back());
        const auto cleanStr = strProfLoc.substr(PROFILING_PREFIX_SIZE);
        std::stringstream sstream(cleanStr);
        std::string token;
        while (std::getline(sstream, token, '_')) {
            if (values.size() < NUM_MANDATORY_FIELDS) {
                values.push_back(std::stoull(token));
            } else {
                maybeTileId = token == "<NONE>" ? Optional<size_t>() : std::stoull(token);
                break;
            }
        }
        const auto rPos = sstream.tellg();
        VPUX_THROW_UNLESS(rPos == -1, "Location should be parsed completely, but remainder left: {0}",
                          sstream.str().substr(rPos));
    } else {
        VPUX_THROW("Profiling should be fused loc");
    }

    VPUX_THROW_UNLESS(values.size() == 3, "Profiling metadata is not what was expected for operation {0}", loc);

    return {values[0], values[1], values[2], maybeTileId};
}

// Update already existing profiling metadata which is a suffix task Loc setting with new
// offset. This is to be used when ActShave task with multiple SwKernelRun ops is being unrolled
mlir::Location getUpdatedActShaveProfilingLoc(mlir::Location loc, size_t tileId) {
    size_t ddrOffset, clusterSize, inClusterOffset;
    std::tie(ddrOffset, clusterSize, inClusterOffset, std::ignore) = parseActShaveProfilingOffsets(loc);
    auto fusedLoc = loc.dyn_cast<mlir::FusedLoc>();
    // Loc has profiling information included at the end. Remove it and recreate with updates offset within cluster
    const auto reducedLoc = mlir::FusedLoc::get(loc->getContext(), fusedLoc.getLocations().drop_back());
    return appendLoc(reducedLoc,
                     createActShaveProfilingLocSuffix(ddrOffset, clusterSize, inClusterOffset + tileId, tileId));
}

mlir::IntegerType getActShaveProfilingElementType(mlir::MLIRContext* ctx) {
    return getUInt32Type(ctx);
}

BaseActShaveProfiler::BaseActShaveProfiler(unsigned clustersAmount, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                           vpux::IndexedSymbolAttr memKindAttr, mlir::FuncOp netFunc, vpux::Logger& log,
                                           std::shared_ptr<NameUniqifier> uniqifier)
        : _clustersAmount(clustersAmount),
          _profilingWorkloadSize(VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES),
          _profilingElementSize(VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES /
                                sizeof(uint32_t)),  // How many DWords are needed to store one workload
          _profilingBufferSizes({0}),
          _builder(builder),
          _ctx(ctx),
          _netFunc(netFunc),
          _memKindAttr(memKindAttr),
          _log(log),
          _uniqifier(uniqifier) {
}

// Get amount of memory needed to store profiling data of all ActShave tasks in the model
unsigned BaseActShaveProfiler::getRequiredDdrMemory() const {
    unsigned swTasksAmount =
            std::accumulate(_swTaskSignatures.begin(), _swTaskSignatures.end(), 0, [](const auto& a, const auto& b) {
                return a + b._maxSubTasks;
            });
    return swTasksAmount * _clustersAmount * _profilingElementSize;
}

// Go over all SwKernelOps and store required information about those tasks like required size of
// profiling buffer or size of profiling buffer instances
void BaseActShaveProfiler::scheduleTask(VPUIP::SwKernelOp swOp) {
    const auto taskSignature = getTaskSignature(swOp);

    // How many elements are needed to store profiling data of one task
    const auto maxSwTasks = taskSignature._maxSubTasks;
    const auto requiredMemory = maxSwTasks * _profilingWorkloadSize;
    VPUX_THROW_WHEN(requiredMemory > VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE,
                    "SwKernelOp at '{0}' requires more memory {1} than currently supported. Change  "
                    "HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE.",
                    swOp->getLoc(), requiredMemory);
    _swTaskSignatures.push_back(taskSignature);
    // Trying to reuse last profiling buffer
    const auto currentBufferSize = _profilingBufferSizes.back();
    const auto newBufferSize = currentBufferSize + maxSwTasks;
    bool isTiled = mlir::isa<VPUIP::NCEClusterTilingOp>(swOp->getParentOp());
    _log.trace("Schedule '{0}' operation with '{1}' subtask, op: '{2}'", (isTiled ? "MultiCluster " : "SingleCluster "),
               maxSwTasks, swOp->getLoc());
    // If we can store profiling result of current task in last buffer without exceeding
    // max size - reuse it, otherwise - scheduling one more
    if (newBufferSize * _profilingWorkloadSize > VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE) {
        _profilingBufferSizes.push_back(maxSwTasks);
    } else {
        _profilingBufferSizes.pop_back();
        _profilingBufferSizes.push_back(newBufferSize);
    }
}

// Main function which goes through all identified ActShave ops and based on gathered data recreates
// those operations to have profiling output with proper slot in profiling buffer instance. When profiling
// buffer is full it also inserts CMX2DDR DMA and allocates new profiling buffer
void BaseActShaveProfiler::addProfilingOps(mlir::BlockArgument& profilingDdrResult,
                                           SmallVector<mlir::Value>& clusterResults) {
    // Contains profiling_output of individual swTaskOp and amount of profiled tiles
    ProfilingResults nceProfilingOutputs;
    size_t currentDDROffset = 0;
    mlir::Operation* currentProfilingBuffer = nullptr;
    unsigned currentBufferSize;
    int currentBufferId = -1;
    const auto allocateProfilingBufferCMX = [&]() {
        if (_profilingBufferSizes.empty()) {
            return;
        }

        ++currentBufferId;
        currentBufferSize = _profilingBufferSizes.front();
        VPUX_THROW_WHEN(currentBufferSize == 0, "Empty CMXBuffers is not allowed");

        _profilingBufferSizes.pop_front();

        const unsigned totalSizeCMXElements = currentBufferSize * _profilingElementSize * _clustersAmount;
        const auto locationName =
                std::to_string(_clustersAmount) + "_actProfilingSubviewBuffer_" + std::to_string(currentBufferId);

        mlir::OpBuilder::InsertPoint lastInsertionPoint = _builder.saveInsertionPoint();
        _builder.setInsertionPointAfter(&_netFunc.getBody().front().front());

        currentProfilingBuffer = createAllocationOp(totalSizeCMXElements, locationName);

        _builder.restoreInsertionPoint(lastInsertionPoint);
    };

    const auto flushCMX2DDR = [&]() {
        if (nceProfilingOutputs.empty() || currentProfilingBuffer == nullptr) {
            return;
        }
        auto copyToDDRResult =
                copyToDdr(nceProfilingOutputs, currentProfilingBuffer, currentDDROffset, profilingDdrResult);
        clusterResults.push_back(copyToDDRResult);

        auto flushedTasksAmount = countTasks(nceProfilingOutputs);
        currentDDROffset += flushedTasksAmount;

        nceProfilingOutputs.clear();
    };

    size_t inClusterOffset = 0;
    // Allocate first buffer for storing profiling results
    allocateProfilingBufferCMX();
    for (auto& swTaskSignature : _swTaskSignatures) {
        auto swTaskOp = swTaskSignature._task;

        auto* insertionPoint = swTaskOp.getOperation();
        auto nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swTaskOp->getParentOp());
        // In case NCE task is wrapped with NCEClusterTiling then inserting should be done
        // at NCEClusterTiling op level and not inside it where NCEClusterTask op is
        if (nceClusterTilingOp) {
            insertionPoint = nceClusterTilingOp.getOperation();
        }
        _builder.setInsertionPoint(insertionPoint);

        const unsigned tasksAmount = swTaskSignature._maxSubTasks * _clustersAmount;
        auto profilingSamplesInCMX = countTasks(nceProfilingOutputs);
        const auto expectedCMXMemoryUsage = (profilingSamplesInCMX + tasksAmount) * _profilingWorkloadSize;
        // If couldnt place current task in the end of cmx buffer flushing all previous tasks to DDR
        // expectedCMXMemoryUsage counts size for all clusters, while HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE only
        // for one so, need to align them for comparison
        if (expectedCMXMemoryUsage > VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE * _clustersAmount) {
            flushCMX2DDR();  // Flush current CMX content to DDR
            profilingSamplesInCMX = 0;
            inClusterOffset = 0;
            allocateProfilingBufferCMX();  // Allocate next CMX buffer
        }

        auto subView = getViewToBuffer(currentProfilingBuffer, profilingSamplesInCMX, tasksAmount);

        // If we have only one tile - we already know his index, otherwise setting None
        Optional<size_t> maybeTileId = swTaskSignature._maxSubTasks == 1 ? 0 : Optional<size_t>();
        const auto profilingMeta =
                createActShaveProfilingLocSuffix(currentDDROffset, currentBufferSize, inClusterOffset, maybeTileId);
        const auto loc = appendLoc(_uniqifier->getUniqueLoc(swTaskOp->getLoc()), profilingMeta);

        auto profilingOutput = replaceOpWithProfiledOp(swTaskOp, subView, loc);

        inClusterOffset += swTaskSignature._maxSubTasks;

        nceProfilingOutputs.push_back({profilingOutput, tasksAmount});
    }
    flushCMX2DDR();
}

SWTaskSignature BaseActShaveProfiler::getTaskSignature(VPUIP::SwKernelOp swOp) const {
    auto numOfProfiledTasks = getNumProfiledTasks(swOp);
    return {swOp, numOfProfiledTasks, {numOfProfiledTasks}};
}

mlir::Type BaseActShaveProfiler::getTimestampType(unsigned tasksAmount) {
    return getMemRefType({_profilingElementSize * tasksAmount}, getActShaveProfilingElementType(_ctx), DimsOrder::C,
                         _memKindAttr);
}

UniformNonTiledActShaveProfiler::UniformNonTiledActShaveProfiler(unsigned clustersAmount, mlir::OpBuilder& builder,
                                                                 mlir::MLIRContext* ctx,
                                                                 vpux::IndexedSymbolAttr memKindAttr,
                                                                 mlir::FuncOp netFunc, vpux::Logger& log,
                                                                 std::shared_ptr<NameUniqifier> uniqifier)
        : BaseActShaveProfiler(clustersAmount, builder, ctx, memKindAttr, netFunc, log, uniqifier) {
}

// Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
// new one needs to be allocated. Type of this alloc is a memref
mlir::Operation* UniformNonTiledActShaveProfiler::createAllocationOp(unsigned totalSizeCMXElements,
                                                                     const std::string& location) {
    auto profBuffType =
            getMemRefType({totalSizeCMXElements}, getActShaveProfilingElementType(_ctx), DimsOrder::C, _memKindAttr);

    _log.trace("Create new allocation op of type - '{0}'", profBuffType);
    return _builder.create<mlir::memref::AllocOp>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                  profBuffType);
}

// Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
// profiling buffer instance is full or there are no more tasks to profile
mlir::Value UniformNonTiledActShaveProfiler::copyToDdr(ProfilingResults profilingResults, mlir::Operation* cmxMemOp,
                                                       size_t& currentDDROffset,
                                                       mlir::BlockArgument& profilingDdrResult) {
    SmallVector<mlir::Value> concatInputs;
    unsigned totalNumElements = 0;
    _log.trace("Insert chunk copy to DDR offset '{0}'", currentDDROffset);
    for (auto& profRes : profilingResults) {
        auto profResult = profRes.first;

        totalNumElements += profRes.second;
        concatInputs.push_back(profResult);
    }

    const auto resultType = mlir::MemRefType::get(
            {static_cast<int64_t>(totalNumElements) * static_cast<int64_t>(_profilingElementSize)},
            getActShaveProfilingElementType(_ctx));

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "actshaveDDR" + std::to_string(currentDDROffset))),
            profilingDdrResult, SmallVector<int64_t>({static_cast<int64_t>(currentDDROffset * _profilingElementSize)}),
            resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(mlir::StringAttr::get(
            _ctx, mlir::StringRef("actshave") + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(currentDDROffset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(
                    _ctx, mlir::StringRef("actshaveProfilingConcat") + std::to_string(currentDDROffset))),
            concatInputs, cmxMemOp->getResult(0));

    return _builder.create<VPUIP::CopyOp>(copyLoc, concatview.output(), subDDR.result());
}

// Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
mlir::Value UniformNonTiledActShaveProfiler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                             unsigned profilingSamplesInCMX, unsigned numTasks) {
    const SmallVector<int64_t> sizes({static_cast<int64_t>(numTasks) * static_cast<int64_t>(_profilingElementSize)});
    int offset = profilingSamplesInCMX * _profilingElementSize;

    _log.trace("Get view to profiling buffer, offset '{0}', size '{1}'", offset, sizes[0]);

    auto subViewLoc =
            appendLoc(currentProfilingBuffer->getLoc(), formatv("_actshaveProfilingSubview_{0}", offset).str());

    auto sub = _builder.create<VPUIP::SubViewOp>(subViewLoc, currentProfilingBuffer->getResult(0),
                                                 SmallVector<int64_t>({static_cast<int>(offset)}), sizes);

    return sub.result();
}

// Replace a Actshave task with new one that has profiling output set
mlir::Value UniformNonTiledActShaveProfiler::replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask,
                                                                     mlir::Value profilingBuffer, mlir::Location loc) {
    _log.trace("Replace op with new profiled task '{0}'", loc);

    SmallVector<mlir::Type> newResultTypes(origSwTask.getResultTypes());
    newResultTypes.push_back(profilingBuffer.getType());

    auto swTask = _builder.create<VPUIP::SwKernelOp>(loc, origSwTask.inputs(), origSwTask.output_buffs(),
                                                     profilingBuffer, origSwTask.kernelFunction(),
                                                     origSwTask.tileIndexAttr(), origSwTask.stridesAttr());

    swTask.getRegion().takeBody(origSwTask.getRegion());

    origSwTask->replaceAllUsesWith(swTask.results());

    return swTask.profiling_output();
}

VPUIP::DistributedBufferType NCETiledActShaveProfiler::getDistributedBufferType(unsigned totalElements) {
    const auto layout = mlir::AffineMapAttr::get(DimsOrder::C.toAffineMap(_ctx));

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(_ctx, VPU::DistributionMode::SEGMENTED);
    const SmallVector<uint64_t> tiles = {_clustersAmount};
    const auto numTiles = getIntArrayAttr(_ctx, tiles);
    const auto numClusters = getIntAttr(_ctx, _clustersAmount);
    const auto memKindAttr = IndexedSymbolAttr::get(_memKindAttr.getLeafNameAttr());
    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTiles, nullptr, nullptr,
                                                                 nullptr, numClusters, nullptr, _ctx);
    return VPUIP::DistributedBufferType::get(_ctx, {totalElements}, getActShaveProfilingElementType(_ctx), layout,
                                             memKindAttr, distributedTensorAttr);
}

NCETiledActShaveProfiler::NCETiledActShaveProfiler(unsigned clustersAmount, mlir::OpBuilder& builder,
                                                   mlir::MLIRContext* ctx, vpux::IndexedSymbolAttr memKindAttr,
                                                   mlir::FuncOp netFunc, vpux::Logger& log,
                                                   std::shared_ptr<NameUniqifier> uniqifier)
        : BaseActShaveProfiler(clustersAmount, builder, ctx, memKindAttr, netFunc, log, uniqifier) {
}

// Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
// new one needs to be allocated. Type of this alloc is a DistributedBufferType
mlir::Operation* NCETiledActShaveProfiler::createAllocationOp(unsigned totalSizeCMXElements,
                                                              const std::string& location) {
    const auto bufferType = getDistributedBufferType(totalSizeCMXElements);
    _log.trace("Create new allocation op of type - '{0}'", bufferType);
    return _builder.create<VPURT::AllocDistributed>(mlir::NameLoc::get(mlir::StringAttr::get(_ctx, location)),
                                                    bufferType, nullptr, nullptr);
}

// Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
// profiling buffer instance is full or there are no more tasks to profile
mlir::Value NCETiledActShaveProfiler::copyToDdr(ProfilingResults profilingResults, mlir::Operation* cmxMemOp,
                                                size_t& currentDDROffset, mlir::BlockArgument& profilingDdrResult) {
    SmallVector<mlir::Value> concatInputs;
    unsigned totalNumElements = 0;

    _log.trace("Insert chunk copy to DDR offset '{0}'", currentDDROffset);
    for (auto& profRes : profilingResults) {
        auto profResult = profRes.first;

        totalNumElements += profRes.second;

        if (profResult.getType().isa<mlir::MemRefType>()) {
            // Result is a plain memref, need to cast back to DistributedBuffer
            auto distType = getDistributedBufferType(profRes.second * _profilingElementSize);
            auto viewLoc = appendLoc(profResult.getLoc(), "_view_cast_to_distributed");
            auto viewOp = _builder.create<VPUIP::ViewOp>(viewLoc, distType, profResult);
            concatInputs.push_back(viewOp.result());
        } else {
            concatInputs.push_back(profResult);
        }
    }

    const auto resultType = mlir::MemRefType::get(
            {static_cast<int64_t>(totalNumElements) * static_cast<int64_t>(_profilingElementSize)},
            getActShaveProfilingElementType(_ctx));

    auto subDDR = _builder.create<VPUIP::SubViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(_ctx, "actshaveDDR" + std::to_string(currentDDROffset))),
            profilingDdrResult, SmallVector<int64_t>({static_cast<int64_t>(currentDDROffset * _profilingElementSize)}),
            resultType.getShape());

    // Create DMA from CMX to Profiling Output
    auto copyLoc = mlir::NameLoc::get(mlir::StringAttr::get(
            _ctx, mlir::StringRef("actshave") + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(currentDDROffset)));
    auto concatview = _builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(
                    _ctx, mlir::StringRef("actshaveProfilingConcat") + std::to_string(currentDDROffset))),
            concatInputs, cmxMemOp->getResult(0));

    SmallVector<mlir::Value> inputsOutputOperands{concatview.output(), subDDR.result()};

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };

    return _builder
            .create<VPUIP::NCEClusterTilingOp>(copyLoc, subDDR.result().getType(), inputsOutputOperands, bodyBuilder)
            .getResult(0);
}

// Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
mlir::Value NCETiledActShaveProfiler::getViewToBuffer(mlir::Operation* currentProfilingBuffer,
                                                      unsigned profilingSamplesInCMX, unsigned numTasks) {
    const SmallVector<int64_t> sizes({static_cast<int64_t>(numTasks) * static_cast<int64_t>(_profilingElementSize)});
    int offset = profilingSamplesInCMX * _profilingElementSize / _clustersAmount;

    _log.trace("Get view to profiling buffer, offset '{0}', size '{1}'", offset, sizes[0]);

    auto subViewLoc =
            appendLoc(currentProfilingBuffer->getLoc(), formatv("_actshaveProfilingSubview_{0}", offset).str());

    auto sub = _builder.create<VPUIP::SubViewOp>(subViewLoc, currentProfilingBuffer->getResult(0),
                                                 SmallVector<int64_t>({static_cast<int64_t>(offset)}), sizes);

    return sub.result();
}

// Replace a Actshave task with new one that has profiling output set. If this task is not multiclustered
// then additional cast (ViewOp) is inserted for profiling slot to maintain type compatibility
mlir::Value NCETiledActShaveProfiler::replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask, mlir::Value profilingBuffer,
                                                              mlir::Location loc) {
    _log.trace("Replace op with new profiled task '{0}'", loc);

    auto profilingSlot = profilingBuffer;

    auto nceClusterTilingOp = origSwTask->getParentOfType<VPUIP::NCEClusterTilingOp>();
    // In case NCE task is wrapped with NCEClusterTiling then inserting should be done
    // at NCEClusterTiling op level and not inside it where NCEClusterTask op is
    if (nceClusterTilingOp) {
        _builder.setInsertionPoint(nceClusterTilingOp.getOperation());
    } else {
        auto viewOpName = appendLoc(loc, "_view_cast");
        auto viewOp = _builder.create<VPUIP::ViewOp>(viewOpName, getTimestampType(1), profilingBuffer);
        profilingSlot = viewOp.result();
    }

    auto swTask = _builder.create<VPUIP::SwKernelOp>(loc, origSwTask.inputs(), origSwTask.output_buffs(), profilingSlot,
                                                     origSwTask.kernelFunction(), origSwTask.tileIndexAttr(),
                                                     origSwTask.stridesAttr());

    // Adjust profiling output to a compact type since later it will be wrapped in NCEClusterTiling
    if (nceClusterTilingOp) {
        auto distType = profilingSlot.getType().dyn_cast<VPUIP::DistributedBufferType>();
        swTask.profiling_output().setType(distType.getCompactType());
    }

    swTask.getRegion().takeBody(origSwTask.getRegion());
    origSwTask->replaceAllUsesWith(swTask.results());

    auto profilingOutput = swTask.profiling_output();

    // In case original ActShaveTask was wrapped with NCEClusterTiling then new ActShaveTask
    // with additional profiling output should also be wrapped with NCEClusterTiling op whose
    // list of operands and results were extended for profiling buffer
    if (nceClusterTilingOp) {
        _log.nest().trace("Wrap task with NCEClusterTiling");
        // Operands of new NCEClusterTilingOp will be extended with profiling buffer
        SmallVector<mlir::Value> newNceClusterTilingOperands(nceClusterTilingOp->getOperands());
        newNceClusterTilingOperands.push_back(swTask.profiling_data());

        // Result of new NCEClusterTilingOp will be extended with profiling result
        SmallVector<mlir::Type> newNceClusterTilingResultTypes(nceClusterTilingOp->getResultTypes());
        newNceClusterTilingResultTypes.push_back(swTask.profiling_data().getType());

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            std::ignore = loc;
            mlir::BlockAndValueMapping mapper;

            auto origArguments = nceClusterTilingOp.body().front().getArguments();

            // Map original NCEClusterTiling argument to new corresponding operands and map
            // profiling buffer to last operand
            mapper.map(origArguments, newOperands.take_front(nceClusterTilingOp->getOperands().size()));
            mapper.map(swTask.profiling_data(), newOperands.take_back(1).front());

            builder.clone(*swTask.getOperation(), mapper);
        };

        auto newNceClusterTilingOp = _builder.create<VPUIP::NCEClusterTilingOp>(
                nceClusterTilingOp->getLoc(), newNceClusterTilingResultTypes, newNceClusterTilingOperands, bodyBuilder);

        // Replace all uses of old NCEClusterTiling op with new one
        // except newly added profiling output
        auto newResults = newNceClusterTilingOp->getResults().drop_back(1);
        nceClusterTilingOp->replaceAllUsesWith(newResults);
        swTask->erase();

        // Set new insertion point back at new NCEClusterTiling level
        _builder.setInsertionPointAfter(newNceClusterTilingOp);

        // Store information about profiling result which later is concatenated with rest of profiling data
        // and copied from buffer in CMX to DDR
        profilingOutput = newNceClusterTilingOp.getResult(newNceClusterTilingOp->getNumResults() - 1);
    }

    return profilingOutput;
}

}  // namespace vpux

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/strings.hpp"

using namespace vpux;

namespace {

//
//  DMATaskProfilingAfterBarrierSchedPass
//

class DMATaskProfilingAfterBarrierSchedPass final :
        public VPUIP::DMATaskProfilingAfterBarrierSchedBase<DMATaskProfilingAfterBarrierSchedPass> {
public:
    explicit DMATaskProfilingAfterBarrierSchedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    mlir::Operation* getDmaTask(VPURT::TaskOp taskOp);
    bool isDmaTask(VPURT::TaskOp taskOp);
    int64_t getDMAPortValue(VPURT::TaskOp taskOp);
    uint32_t getHwProfAddress(VPU::ArchKind arch);
    vpux::NDTypeInterface getTimestampType(mlir::MLIRContext* ctx, VPU::ArchKind arch);
    mlir::NameLoc getProfTaskLoc(mlir::MLIRContext* ctx, unsigned dmaId, mlir::Location taskOpLoc, bool after);
    VPURT::TaskOp createProfTask(mlir::OpBuilder& builder, mlir::ValueRange updateBarriers,
                                 mlir::ValueRange waitBarriers, int64_t port, size_t address, mlir::NameLoc loc);

    void createCmx2DdrProfDma(mlir::OpBuilder& builder, mlir::ValueRange updateBarriers, int64_t port,
                              size_t srcCmxAddr, size_t dstDdrAddr, size_t sizeBytes);

    vpux::NDTypeInterface _timestampTypeSingleSlot = nullptr;
    vpux::NDTypeInterface _profilingBufferType = nullptr;
    vpux::NDTypeInterface _hwTimerType = nullptr;
    VPURT::BufferSection _profBufSection = VPURT::BufferSection::DDR;
    int64_t _profBufSectionIndex = 0;
    uint32_t _hwAddr = 0;
    int64_t _profOutputId = 0;
};

uint32_t DMATaskProfilingAfterBarrierSchedPass::getHwProfAddress(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return VPUIP::HW_TIMER_ABSOLUTE_ADDR_30XX;
    case VPU::ArchKind::VPUX37XX:
        return VPUIP::HW_TIMER_ABSOLUTE_ADDR_37XX;
    default:
        VPUX_THROW("Unsuported architecture for TimestampRewrite");
    }
}

vpux::NDTypeInterface DMATaskProfilingAfterBarrierSchedPass::getTimestampType(mlir::MLIRContext* ctx,
                                                                              VPU::ArchKind arch) {
    // DMA profiling data is stored in CMX slice 0
    auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, memKindAttr);
    case VPU::ArchKind::VPUX37XX:
        return getMemRefType(ShapeRef({1}), getUInt64Type(ctx), DimsOrder::C, memKindAttr);
    default:
        VPUX_THROW("Not supported architecture");
    }
}

int64_t DMATaskProfilingAfterBarrierSchedPass::getDMAPortValue(VPURT::TaskOp taskOp) {
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();

    VPUX_THROW_WHEN(mlir::isa<VPUIP::NCEClusterTilingOp>(wrappedTaskOp),
                    "NCEClusterTiling is not expected at this stage of compilation");

    return vpux::getDMAPortValue(wrappedTaskOp);
}

bool DMATaskProfilingAfterBarrierSchedPass::isDmaTask(VPURT::TaskOp taskOp) {
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();

    VPUX_THROW_WHEN(mlir::isa<VPUIP::NCEClusterTilingOp>(wrappedTaskOp),
                    "NCEClusterTiling is not expected at this stage of compilation");

    return mlir::isa_and_nonnull<VPUIP::ProfiledDMAOpInterface>(wrappedTaskOp);
}

mlir::NameLoc DMATaskProfilingAfterBarrierSchedPass::getProfTaskLoc(mlir::MLIRContext* ctx, unsigned dmaId,
                                                                    mlir::Location taskOpLoc, bool after) {
    std::string curTaskName;
    curTaskName = stringifyLocation(taskOpLoc);
    return mlir::NameLoc::get(mlir::StringAttr::get(
            ctx, curTaskName + ((!after) ? (dmaId == 0 ? PROFILING_DMA_BEGIN_SUFFIX : PROFILING_DMA_TASK_BEGIN_SUFFIX)
                                         : (PROFILING_DMA_TASK_END_SUFFIX + std::to_string(dmaId - 1) + "_" +
                                            std::to_string(dmaId / 2 + 1)))));
}

VPURT::TaskOp DMATaskProfilingAfterBarrierSchedPass::createProfTask(mlir::OpBuilder& builder,
                                                                    mlir::ValueRange waitBarriers,
                                                                    mlir::ValueRange updateBarriers, int64_t port,
                                                                    size_t address, mlir::NameLoc loc) {
    _log.trace("createProfTask: port '{0}' cmxAddress '{1}' loc '{2}'", port, address, loc);

    // Create declaration for source buffer which corresponds to HW register with free-running counter
    auto hwRegOp = builder.create<VPURT::DeclareBufferOp>(loc, _hwTimerType, VPURT::BufferSection::Register, _hwAddr);

    // Create declaration for destination buffer in CMX where timestamp is to be stored
    auto profBufOp = builder.create<VPURT::DeclareBufferOp>(loc, _timestampTypeSingleSlot, _profBufSection,
                                                            makeArrayRef(_profBufSectionIndex), address);

    // Insert profiling DMA wrapped in TaskOp with proper barriers settings
    const auto profDMA = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, waitBarriers, updateBarriers, loc,
                                                               hwRegOp.buffer(), profBufOp.buffer(), port);

    return profDMA->getParentOfType<VPURT::TaskOp>();
}

void DMATaskProfilingAfterBarrierSchedPass::createCmx2DdrProfDma(mlir::OpBuilder& builder,
                                                                 mlir::ValueRange updateBarriers, int64_t port,
                                                                 size_t srcCmxAddr, size_t dstDdrAddr,
                                                                 size_t sizeBytes) {
    auto* ctx = builder.getContext();
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(
            ctx, mlir::StringRef("dma") + PROFILING_CMX_2_DDR_OP_NAME + std::to_string(dstDdrAddr)));

    _log.trace("Create CMX to DDR profiling buffer copy, port '{0}', cmxAddress '{1}' ddrAddress '{2}', "
               "size '{3}' loc '{4}'",
               port, srcCmxAddr, dstDdrAddr, sizeBytes, loc);

    auto numOfElements = static_cast<int64_t>(sizeBytes / Byte(_timestampTypeSingleSlot.getElemTypeSize()).count());
    auto profilingBufferType = _timestampTypeSingleSlot.changeShape(ShapeRef({numOfElements}));

    // Create declaration for source buffer which corresponds to HW register with free-running counter
    auto srcBufProfCmxOp = builder.create<VPURT::DeclareBufferOp>(loc, profilingBufferType, _profBufSection,
                                                                  makeArrayRef(_profBufSectionIndex), srcCmxAddr);

    // Create declaration for destination in profiling output
    auto profilingOutputType =
            mlir::MemRefType::get(profilingBufferType.getShape().raw(), profilingBufferType.getElementType());
    auto dstBufProfResultOp = builder.create<VPURT::DeclareBufferOp>(
            loc, profilingOutputType, VPURT::BufferSection::ProfilingOutput, _profOutputId, dstDdrAddr);

    // // Insert DMA wrapped in TaskOp with proper barriers settings
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, {}, updateBarriers, loc, srcBufProfCmxOp.buffer(),
                                          dstBufProfResultOp.buffer(), port);
}

void DMATaskProfilingAfterBarrierSchedPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();
    const auto arch = VPU::getArch(module);

    IE::CNNNetworkOp netOp;
    mlir::FuncOp func;
    IE::CNNNetworkOp::getFromModule(module, netOp, func);
    mlir::OpBuilder builder(&func.getBody().front().front());

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    int64_t dmaProfReservedMemSize = 0;
    int64_t dmaProfMemOffset = 0;
    if (auto dmaProfMem = IE::getDmaProfilingReservedMemory(module, VPU::MemoryKind::CMX_NN)) {
        dmaProfReservedMemSize = dmaProfMem.byteSize();
        VPUX_THROW_UNLESS(dmaProfMem.offset().hasValue(), "No offset setting provided");
        dmaProfMemOffset = dmaProfMem.offset().getValue();
    }

    VPUX_THROW_UNLESS(dmaProfReservedMemSize > 0, "No reserved memory for DMA profiling");
    VPUX_THROW_UNLESS((dmaProfReservedMemSize % dmaPortCount) == 0,
                      "Reserved memory for DMA profiling ('{0}') cannot be equally split between ports",
                      dmaProfReservedMemSize);

    const int64_t dmaProfReservedMemSizePerPort = static_cast<int64_t>(dmaProfReservedMemSize / dmaPortCount);

    // Traverse IR and store information about DMA tasks on each port
    SmallVector<SmallVector<VPURT::TaskOp>> dmaTasksPerPort(dmaPortCount);

    // Gather information about DMA tasks and their port so that total number is known
    // before inserting profiling tasks.
    func->walk([&](VPURT::TaskOp taskOp) {
        if (!isDmaTask(taskOp)) {
            return;
        }

        auto taskName = stringifyLocation(taskOp->getLoc());
        // Skip DMAs which are used for handling profiling data or spilling. Such DMAs will not be measured.
        if (taskName.find(PROFILING_CMX_2_DDR_OP_NAME) != std::string::npos ||
            taskName.find(SPILL_READ_OP_NAME_SUFFIX) != std::string::npos ||
            taskName.find(SPILL_WRITE_OP_NAME_SUFFIX) != std::string::npos) {
            return;
        }

        auto portNumber = getDMAPortValue(taskOp);
        dmaTasksPerPort[portNumber].push_back(taskOp);
    });

    size_t totalNumberOfDmas = 0;
    size_t ddrProfilingBufferOffset = 0;
    SmallVector<size_t> cmxProfilingBaseAddress(dmaPortCount);
    SmallVector<size_t> cmxProfilingBufferOffset(dmaPortCount);

    for (int64_t port = 0; port < dmaPortCount; port++) {
        cmxProfilingBaseAddress[port] = dmaProfMemOffset + port * dmaProfReservedMemSizePerPort;
        totalNumberOfDmas += dmaTasksPerPort[port].size();
    }

    // Initialize some common variables which are used when inserting profiling DMAs
    _timestampTypeSingleSlot = getTimestampType(ctx, arch);
    _hwAddr = getHwProfAddress(arch);
    _hwTimerType = _timestampTypeSingleSlot.changeMemSpace(VPU::MemoryKind::Register);
    _profBufSection = VPURT::getBufferSection(_timestampTypeSingleSlot.getMemoryKind());
    auto sectionIndex = _timestampTypeSingleSlot.getMemSpace().getIndex();
    VPUX_THROW_UNLESS(sectionIndex.hasValue(), "Destination buffer without section index value");
    _profBufSectionIndex = sectionIndex.getValue();
    _profOutputId = static_cast<int64_t>(netOp.getProfilingOutputsCount());

    const unsigned timestampSingleSlotSize = Byte(_timestampTypeSingleSlot.getElemTypeSize()).count();
    const auto outputResult = mlir::MemRefType::get({static_cast<unsigned>(2 * totalNumberOfDmas)},
                                                    _timestampTypeSingleSlot.getElementType());

    // Update network output information to have also new dma profiling result
    auto profilingResult = addNewProfilingOutput(ctx, func, netOp, outputResult, "dma");
    auto returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(func.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);
    returnOp.operandsMutable().append(profilingResult);

    // Go through DMA tasks on each port and insert profiling tasks
    // Example:
    // Before pass:
    //  <waitBarrier> -> DMA0 -> <updateBarrier>
    // After pass:
    //  <waitBarrier> -> Prof -> DMA0 -> Prof -> <updateBarrier>
    //
    // When profiling buffer is full or there are no more tasks to profile then
    // DMA that copies profiling buffer to DDR is inserted right after last profiling DMA

    unsigned dmaId = 0;
    for (int portNumber = 0; portNumber < dmaPortCount; portNumber++) {
        _log.trace("Handle DMA profiling on port '{0}', number of tasks to profile: '{1}'", portNumber,
                   dmaTasksPerPort[portNumber].size());
        for (auto i : irange(dmaTasksPerPort[portNumber].size())) {
            auto& taskOp = dmaTasksPerPort[portNumber][i];

            _log.trace("DMA task to profile: '{0}'", taskOp->getLoc());
            _log = _log.nest();

            auto waitBarriers = taskOp.waitBarriers();
            auto updateBarriers = taskOp.updateBarriers();

            // First check if after this task CMX2DDR profiling buffer copy will be needed
            // due to buffer geting full after this task
            auto insertProfCmx2DdrFlag = (dmaProfReservedMemSizePerPort - cmxProfilingBufferOffset[portNumber] ==
                                          2 * timestampSingleSlotSize);
            // Also if this is last DMA to profile on this port then such copy should be done too
            insertProfCmx2DdrFlag = (insertProfCmx2DdrFlag || (i == (dmaTasksPerPort[portNumber].size() - 1)));

            // Handle start time profiling task
            builder.setInsertionPoint(taskOp);
            auto cmxAddress = cmxProfilingBaseAddress[portNumber] + cmxProfilingBufferOffset[portNumber];

            _log.trace("Create task start profiling DMA");
            _log = _log.nest();
            auto profTaskLoc = getProfTaskLoc(ctx, dmaId, taskOp->getLoc(), false);
            createProfTask(builder, waitBarriers, {}, portNumber, cmxAddress, profTaskLoc);
            dmaId++;
            _log = _log.unnest();

            cmxProfilingBufferOffset[portNumber] += timestampSingleSlotSize;

            // Handle end time profiling task
            builder.setInsertionPointAfter(taskOp);
            cmxAddress = cmxProfilingBaseAddress[portNumber] + cmxProfilingBufferOffset[portNumber];

            _log.trace("Create task end profiling DMA");
            _log = _log.nest();
            profTaskLoc = getProfTaskLoc(ctx, dmaId, taskOp->getLoc(), true);
            if (insertProfCmx2DdrFlag) {
                // In case after this task DMA to DDR with profiling buffer is inserted then
                // updateBarriers settings will be pushed to it and not needed here
                createProfTask(builder, {}, {}, portNumber, cmxAddress, profTaskLoc);
            } else {
                createProfTask(builder, {}, updateBarriers, portNumber, cmxAddress, profTaskLoc);
            }
            dmaId++;
            _log = _log.unnest();
            cmxProfilingBufferOffset[portNumber] += timestampSingleSlotSize;

            VPUX_THROW_WHEN(static_cast<int64_t>(cmxProfilingBufferOffset[portNumber]) > dmaProfReservedMemSizePerPort,
                            "Profiling buffer beyond allowed space");

            if (insertProfCmx2DdrFlag) {
                // Once profiling buffer instance is full or there are no more tasks to profile
                // insert DMA which copies buffer to ProfilingOutput at proper offset
                auto size = cmxProfilingBufferOffset[portNumber];

                createCmx2DdrProfDma(builder, updateBarriers, portNumber, cmxProfilingBaseAddress[portNumber],
                                     ddrProfilingBufferOffset, size);

                ddrProfilingBufferOffset += size;
                cmxProfilingBufferOffset[portNumber] = 0;
            }

            // Remove barriers from profiled task as its execution will be guarded by
            // newly inserted profiling DMAs before and after the task
            taskOp.waitBarriersMutable().clear();
            taskOp.updateBarriersMutable().clear();

            _log = _log.unnest();
        }
    }
}

}  // namespace

//
// createDMATaskProfilingAfterBarrierSchedPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMATaskProfilingAfterBarrierSchedPass(Logger log) {
    return std::make_unique<DMATaskProfilingAfterBarrierSchedPass>(log);
}

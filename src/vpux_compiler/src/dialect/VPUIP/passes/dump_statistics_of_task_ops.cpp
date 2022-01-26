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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

//
// DumpStatisticsOfTaskOpsPass
//

class DumpStatisticsOfTaskOpsPass final : public VPUIP::DumpStatisticsOfTaskOpsBase<DumpStatisticsOfTaskOpsPass> {
public:
    explicit DumpStatisticsOfTaskOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DumpStatisticsOfTaskOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    _log.info("VPUIP tasks statistics:");

    llvm::DenseSet<mlir::OperationName> dpuOperations{
            mlir::OperationName(VPUIP::ConvolutionUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::PoolingUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::EltwiseUPAOp::getOperationName(), &ctx)};

    llvm::DenseMap<mlir::OperationName, size_t> taskMap;
    size_t Cmx2CmxDmas = 0;
    size_t Ddr2DdrDmas = 0;
    size_t Ddr2CmxDmas = 0;
    size_t Cmx2DdrDmas = 0;

    func->walk([&](VPUIP::TaskOpInterface op) {
        taskMap[op->getName()]++;
    });

    func->walk([&](VPURT::ConfigureBarrierOp op) {
        taskMap[op->getName()]++;
    });

    func->walk([&](VPUIP::NNDMAOp dmaOp) {
        const auto srcMemory = VPU::getMemoryKind(dmaOp.input().getType().cast<mlir::MemRefType>());
        const auto dstMemory = VPU::getMemoryKind(dmaOp.output_buff().getType().cast<mlir::MemRefType>());

        if (srcMemory == VPU::MemoryKind::CMX_NN && dstMemory == VPU::MemoryKind::DDR) {
            Cmx2DdrDmas++;
        } else if (srcMemory == VPU::MemoryKind::DDR && dstMemory == VPU::MemoryKind::CMX_NN) {
            Ddr2CmxDmas++;
        } else if (srcMemory == VPU::MemoryKind::CMX_NN && dstMemory == VPU::MemoryKind::CMX_NN) {
            Cmx2CmxDmas++;
        } else if (srcMemory == VPU::MemoryKind::DDR && dstMemory == VPU::MemoryKind::DDR) {
            Ddr2DdrDmas++;
        }
    });

    for (auto& taskOp : taskMap) {
        _log.nest().info("{0} - {1} ops", taskOp.first, taskOp.second);
        if (taskOp.first.getStringRef() == VPUIP::NNDMAOp::getOperationName()) {
            if (Cmx2DdrDmas > 0) {
                _log.nest(2).info("{0} CMX2DDR - {1} ops", taskOp.first, Cmx2DdrDmas);
            }
            if (Ddr2CmxDmas > 0) {
                _log.nest(2).info("{0} DDR2CMX - {1} ops", taskOp.first, Ddr2CmxDmas);
            }
            if (Cmx2CmxDmas > 0) {
                _log.nest(2).info("{0} CMX2CMX - {1} ops", taskOp.first, Cmx2CmxDmas);
            }
            if (Ddr2DdrDmas > 0) {
                _log.nest(2).info("{0} DDR2DDR - {1} ops", taskOp.first, Ddr2DdrDmas);
            }
        }

        if (VPU::getCompilationMode(func) == VPU::CompilationMode::ReferenceSW) {
            continue;
        }

        if (dpuOperations.contains(taskOp.first)) {
            _log.nest().warning("'{0}' was not converted to 'VPUIP.NCETask'", taskOp.first);
        }
    }
}

}  // namespace

//
// createDumpStatisticsOfTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDumpStatisticsOfTaskOpsPass(Logger log) {
    return std::make_unique<DumpStatisticsOfTaskOpsPass>(log);
}

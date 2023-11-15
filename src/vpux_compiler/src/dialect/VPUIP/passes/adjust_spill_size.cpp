//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

class AdjustSpillSizePass final : public VPUIP::AdjustSpillSizeBase<AdjustSpillSizePass> {
public:
    explicit AdjustSpillSizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    int64_t getAdjustedSpillBufferSize(vpux::NDTypeInterface type);
    void updateSpillWrite(VPUIP::CopyOp copyOp);
    void updateSpillRead(VPUIP::CopyOp copyOp);

    mlir::DenseMap<int64_t, mlir::Type> _spillIdAndTypeMap;
};

int64_t AdjustSpillSizePass::getAdjustedSpillBufferSize(vpux::NDTypeInterface type) {
    // In worst case scenario depending on the content of activation, its final size after
    // compression might be bigger than original size. Compiler before performing DDR
    // allocation needs to adjust required size by this buffer
    // Formula from HAS is following:
    //   DTS = X * Y * Z * (element size in bytes)
    //   denseSize = (DTS * (65/64)) + 1
    //   DDR Allocation (32B aligned) = denseSize + ( (denseSize % 32) ? (32 – (denseSize % 32) : 0)
    auto worstCaseSize = static_cast<int64_t>(type.getTotalAllocSize().count() * 65 / 64) + 1;
    if (worstCaseSize % VPUIP::ACT_COMPRESSION_BUF_SIZE_ALIGNMENT) {
        worstCaseSize +=
                VPUIP::ACT_COMPRESSION_BUF_SIZE_ALIGNMENT - worstCaseSize % VPUIP::ACT_COMPRESSION_BUF_SIZE_ALIGNMENT;
    }

    return worstCaseSize;
}

// Update spill-write op have new type that reflects desired size of allocation
void AdjustSpillSizePass::updateSpillWrite(VPUIP::CopyOp copyOp) {
    _log.trace("Spill Write op identified - '{0}'", copyOp->getLoc());

    auto* ctx = copyOp->getContext();
    auto spillBuffer = copyOp.output_buff();

    auto asyncOp = copyOp->getParentOfType<mlir::async::ExecuteOp>();
    auto clusterOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    mlir::BlockArgument blockArg;
    if (clusterOp) {
        asyncOp = clusterOp->getParentOfType<mlir::async::ExecuteOp>();
        blockArg = copyOp.output_buff().cast<mlir::BlockArgument>();
        spillBuffer = clusterOp->getOperand(blockArg.getArgNumber());
    }
    VPUX_THROW_WHEN(asyncOp == nullptr, "No async execute identified for given SpillWrite CopyOp - '{0}'",
                    copyOp->getLoc());

    auto spillAllocOp = spillBuffer.getDefiningOp();
    VPUX_THROW_UNLESS((mlir::isa<mlir::memref::AllocOp, VPURT::Alloc>(spillAllocOp)),
                      "Unexpected allocation operation for spill buffer - '{0}'", spillAllocOp);

    auto spillAllocOpResult = spillAllocOp->getResult(0);

    // Allocation size for activation spill should be adjusted. This is because later those spills will be converted
    // to compressed DMA and in worst case scenario size of compressed activation can be in fact bigger than
    // original buffer. To prevent from memory corruption spill buffer size needs to be adjusted

    auto spillType = spillAllocOpResult.getType().cast<vpux::NDTypeInterface>();

    auto spillTypeMemref = spillType.dyn_cast<mlir::MemRefType>();
    VPUX_THROW_WHEN(spillTypeMemref == nullptr, "Expected memref type for spilled buffer but got - '{0}'", spillType);

    const auto layout = spillTypeMemref.getLayout();

    const auto orderAttr = mlir::AffineMapAttr::get(spillType.getDimsOrder().toAffineMap(ctx));
    mlir::ArrayAttr stridesAttr = nullptr;
    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr;
    VPUIP::CompressionSchemeAttr compressionSchemeAttr = nullptr;

    if (const auto memRefAttr = layout.dyn_cast<VPUIP::MemRefAttr>()) {
        stridesAttr = memRefAttr.strides();
        swizzlingSchemeAttr = memRefAttr.swizzlingScheme();
        compressionSchemeAttr = memRefAttr.compressionScheme();
    }

    mlir::IntegerAttr allocSizeAttr = getIntAttr(ctx, getAdjustedSpillBufferSize(spillType));

    const auto newLayoutAttr = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, swizzlingSchemeAttr,
                                                      compressionSchemeAttr, allocSizeAttr, ctx);

    mlir::MemRefType::Builder builder(spillTypeMemref);
    builder.setLayout(newLayoutAttr.cast<mlir::MemRefLayoutAttrInterface>());

    auto newSpillTypeMemref = static_cast<mlir::MemRefType>(builder);
    _log.nest().trace("New spilled buffer type - '{0}'", newSpillTypeMemref);

    // Update type of allocation operation and also result types for spill write operation
    spillAllocOpResult.setType(newSpillTypeMemref);
    copyOp->getResult(0).setType(newSpillTypeMemref);

    if (clusterOp) {
        blockArg.setType(newSpillTypeMemref);
        clusterOp->getResult(0).setType(newSpillTypeMemref);
    }

    auto asyncSpillResult = asyncOp.results()[0];
    asyncSpillResult.setType(mlir::async::ValueType::get(newSpillTypeMemref));

    _spillIdAndTypeMap[copyOp.spillId().value()] = newSpillTypeMemref;
}

// Based on corresponding spill-write op this function will update
// spill-read operation to have new type that reflects desired size of allocation
void AdjustSpillSizePass::updateSpillRead(VPUIP::CopyOp copyOp) {
    _log.trace("Spill Read op identified - '{0}'", copyOp->getLoc());

    auto spillId = copyOp.spillId().value();
    VPUX_THROW_UNLESS(_spillIdAndTypeMap.find(spillId) != _spillIdAndTypeMap.end(),
                      "No matching spill write was located before");

    auto newSpillTypeMemref = _spillIdAndTypeMap[spillId];

    auto spillBuffer = copyOp.input();

    // Update type block args of operations which wrap spill read copy op
    auto asyncOp = copyOp->getParentOfType<mlir::async::ExecuteOp>();
    auto clusterOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();

    if (clusterOp) {
        asyncOp = clusterOp->getParentOfType<mlir::async::ExecuteOp>();
        auto blockArg = copyOp.input().dyn_cast<mlir::BlockArgument>();

        spillBuffer = clusterOp->getOperand(blockArg.getArgNumber());

        blockArg.setType(newSpillTypeMemref);
    }

    VPUX_THROW_WHEN(asyncOp == nullptr, "No async execute identified for given SpillRead CopyOp - '{0}'",
                    copyOp->getLoc());

    if (auto blockArg = spillBuffer.dyn_cast<mlir::BlockArgument>()) {
        blockArg.setType(newSpillTypeMemref);
    }
}

void AdjustSpillSizePass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPUIP::CopyOp copyOp) {
        if (!copyOp.spillId().has_value()) {
            return;
        }

        const auto inType = copyOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto outType = copyOp.output().getType().cast<vpux::NDTypeInterface>();

        if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN && outType.getMemoryKind() == VPU::MemoryKind::DDR) {
            updateSpillWrite(copyOp);
        } else if (inType.getMemoryKind() == VPU::MemoryKind::DDR &&
                   outType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            updateSpillRead(copyOp);
        }
    });
}

}  // namespace

//
// createAdjustSpillSizePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAdjustSpillSizePass(Logger log) {
    return std::make_unique<AdjustSpillSizePass>(log);
}

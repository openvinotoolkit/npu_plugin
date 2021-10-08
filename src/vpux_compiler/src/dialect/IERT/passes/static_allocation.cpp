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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseSet.h>

using namespace vpux;

namespace {

//
// LinearScanHandler
//

class LinearScanHandler final {
public:
    explicit LinearScanHandler(AddressType defaultAlignment = 1): _defaultAlignment(defaultAlignment) {
    }

public:
    void markAsDead(mlir::Value val) {
        _aliveValues.erase(val);
    }
    void markAsAlive(mlir::Value val) {
        _aliveValues.insert(val);
    }

    auto maxAllocatedSize() const {
        return _maxAllocatedSize;
    }

public:
    bool isAlive(mlir::Value val) const {
        return _aliveValues.contains(val);
    }

    static bool isFixedAlloc(mlir::Value val) {
        return val.getDefiningOp<IERT::StaticAllocOp>() != nullptr;
    }

    static AddressType getSize(mlir::Value val) {
        const auto type = val.getType().dyn_cast<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "StaticAllocation can work only with MemRef Type, got '{0}'", val.getType());

        const Byte totalSize = getTypeTotalSize(type);
        return checked_cast<AddressType>(totalSize.count());
    }

    AddressType getAlignment(mlir::Value val) const {
        if (auto allocOp = val.getDefiningOp<mlir::memref::AllocOp>()) {
            if (auto alignment = allocOp.alignment()) {
                return checked_cast<AddressType>(alignment.getValue());
            }
        }

        return _defaultAlignment;
    }

    AddressType getAddress(mlir::Value val) const {
        if (auto staticAllocOp = val.getDefiningOp<IERT::StaticAllocOp>()) {
            return checked_cast<AddressType>(staticAllocOp.offset());
        }

        const auto it = _valOffsets.find(val);
        VPUX_THROW_UNLESS(it != _valOffsets.end(), "Value '{0}' was not allocated", val);

        return it->second;
    }

    void allocated(mlir::Value val, AddressType addr) {
        VPUX_THROW_UNLESS(addr != InvalidAddress, "Trying to assign invalid address");
        VPUX_THROW_UNLESS(_valOffsets.count(val) == 0, "Value '{0}' was already allocated", val);

        _valOffsets.insert({val, addr});

        const auto endAddr = alignVal<int64_t>(addr + getSize(val), getAlignment(val));
        _maxAllocatedSize = Byte(std::max(_maxAllocatedSize.count(), endAddr));
    }

    void freed(mlir::Value val) {
        markAsDead(val);
    }

    static int getSpillWeight(mlir::Value) {
        VPUX_THROW("Spills are not implemented");
    }

    static bool spilled(mlir::Value) {
        VPUX_THROW("Spills are not implemented");
    }

private:
    mlir::DenseMap<mlir::Value, AddressType> _valOffsets;
    mlir::DenseSet<mlir::Value> _aliveValues;
    AddressType _defaultAlignment = 1;
    Byte _maxAllocatedSize;
};

using LinearScanImpl = LinearScan<mlir::Value, LinearScanHandler>;

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    AllocRewrite(const LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.memref();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));

    _log.trace("Replace with statically allocated VPUIP.DeclareTensorOp (offset = {0})", offset);
    rewriter.replaceOpWithNewOp<IERT::StaticAllocOp>(origOp, val.getType(), offset);

    return mlir::success();
}

//
// StaticAllocationPass
//

class StaticAllocationPass final : public IERT::StaticAllocationBase<StaticAllocationPass> {
public:
    StaticAllocationPass(IERT::AttrCreateFunc memSpaceCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    LinearScanHandler runLinearScan(mlir::FuncOp netFunc, IERT::RunTimeResourcesOp resources);

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

StaticAllocationPass::StaticAllocationPass(IERT::AttrCreateFunc memSpaceCb, Logger log)
        : _memSpaceCb(std::move(memSpaceCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult StaticAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    _memSpace = _memSpaceCb(ctx, memSpaceName.getValue());

    if (_memSpace == nullptr) {
        return mlir::failure();
    }

    return mlir::success();
}

LinearScanHandler StaticAllocationPass::runLinearScan(mlir::FuncOp netFunc, IERT::RunTimeResourcesOp resources) {
    auto& liveRangeInfo = getChildAnalysis<MemLiveRangeInfo>(netFunc);
    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    auto availableMem = resources.getAvailableMemory(_memSpace);
    VPUX_THROW_WHEN(availableMem == nullptr, "The memory space '{0}' is not available", _memSpace);

    const Byte maxMemSize = availableMem.size();
    const uint64_t memDefaultAlignment = 64;  // TODO: extract from run-time resources information?

    LinearScanImpl scan(maxMemSize.count(), memDefaultAlignment);

    const auto allocNewBuffers = [&](const ValueOrderedSet& usedBufs) {
        _log.trace("Locate new buffers");
        _log = _log.nest();

        SmallVector<mlir::Value> newBufs;

        for (auto val : usedBufs) {
            const auto type = val.getType().cast<mlir::MemRefType>();
            if (type.getMemorySpace() != _memSpace) {
                continue;
            }

            _log.trace("Check buffer '{0}'", val);

            if (scan.handler().isAlive(val)) {
                continue;
            }

            _log.nest().trace("This task is the first usage of the buffer, allocate it");

            scan.handler().markAsAlive(val);
            newBufs.push_back(val);
        }

        _log.trace("Alocate memory for the new buffers");
        VPUX_THROW_UNLESS(scan.alloc(newBufs, /*allowSpills*/ false), "Failed to statically allocate '{0}' memory",
                          _memSpace);

        _log = _log.unnest();
    };

    const auto freeDeadBuffers = [&](const ValueOrderedSet& usedBufs, mlir::async::ExecuteOp op) {
        _log.trace("Locate dead buffers");
        _log = _log.nest();

        for (auto val : usedBufs) {
            const auto type = val.getType().cast<mlir::MemRefType>();
            if (type.getMemorySpace() != _memSpace) {
                continue;
            }

            _log.trace("Check buffer '{0}'", val);

            if (liveRangeInfo.eraseUser(val, op) == 0) {
                _log.nest().trace("This bucket is the last usage of the buffer, free it");
                scan.handler().markAsDead(val);
            }
        }

        _log.trace("Free memory for the dead buffers");
        scan.freeNonAlive();

        _log = _log.unnest();
    };

    mlir::async::ExecuteOp prevExecOp;
    for (auto curExecOp : netFunc.getOps<mlir::async::ExecuteOp>()) {
        _log.trace("Process next task at '{0}'", curExecOp->getLoc());
        _log = _log.nest();

        // TODO: remove temporary linearization
        if (prevExecOp != nullptr) {
            _log.trace("Add explicit dependency from '{0}' to '{1}'", prevExecOp->getLoc(), curExecOp->getLoc());
            depsInfo.addDependency(prevExecOp, curExecOp);
        }

        const auto usedBufs = liveRangeInfo.getUsedBuffers(curExecOp);

        allocNewBuffers(usedBufs);
        freeDeadBuffers(usedBufs, curExecOp);

        prevExecOp = curExecOp;

        _log = _log.unnest();
    }

    depsInfo.updateTokenDependencies();

    return scan.handler();
}

void StaticAllocationPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto allocInfo = runLinearScan(netFunc, resources);
    resources.setUsedMemory(_memSpace, allocInfo.maxAllocatedSize());

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().dyn_cast<mlir::MemRefType>();
        return type == nullptr || type.getMemorySpace() != _memSpace;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AllocRewrite>(allocInfo, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IERT::createStaticAllocationPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<StaticAllocationPass>(std::move(memSpaceCb), log);
}

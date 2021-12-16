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

#pragma once

#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/ring_buffer.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/DenseMap.h>

#include <array>
#include <cassert>
#include <cstdint>

namespace vpux {
namespace VPURT {

static constexpr int64_t MAX_DMA_ENGINES = 2;
static constexpr int64_t MAX_NCE_CLUSTERS = 7;

struct BarrierConfig final {
    int64_t realId = -1;
    int64_t nextSameId = -1;
    int64_t producerCount = 0;
    int64_t consumerCount = 0;

    BarrierConfig() = default;

    explicit BarrierConfig(int64_t realId): realId(realId) {
    }
};

struct BarrierUserConfig final {
    uint64_t waitMask = 0;
    uint64_t postMask = 0;
    int64_t startAfter = -1;
    int64_t cleanAfter = -1;
    int64_t virtualDep = -1;
    int64_t count = 0;

    BarrierUserConfig() = default;

    explicit BarrierUserConfig(int64_t virtualDep, int64_t count = 1): virtualDep(virtualDep), count(count) {
    }
};

//
// VirtualDependencyTracker
//

class VirtualDependencyTracker final {
public:
    using Range = std::pair<int64_t, int64_t>;

    struct Dependency final {
        Range consumer;
        Range producer;
    };

public:
    VirtualDependencyTracker(): _deps(1) {
    }

public:
    int64_t add(VPURT::TaskOp taskOp);
    int64_t add(ArrayRef<int64_t> waits, ArrayRef<int64_t> posts);

    int64_t clone(int64_t i);

public:
    bool validId(int64_t i) const {
        return i >= 0 && checked_cast<size_t>(i) < _ids.size();
    }
    bool validDep(int64_t i) const {
        return i >= 0 && checked_cast<size_t>(i) < _deps.size();
    }

    int64_t id(int64_t i) const {
        assert(validId(i));
        return _ids[checked_cast<size_t>(i)];
    }
    int64_t& id(int64_t i) {
        assert(validId(i));
        return _ids[checked_cast<size_t>(i)];
    }

    const Dependency& dep(int64_t i) const {
        assert(validDep(i));
        return _deps[checked_cast<size_t>(i)];
    }
    Dependency& dep(int64_t i) {
        assert(validDep(i));
        return _deps[checked_cast<size_t>(i)];
    }

public:
    void print(Logger log) const;

private:
    SmallVector<int64_t> _ids;
    SmallVector<Dependency> _deps;
};

//
// BarrierSimulator
//

class BarrierSimulator final {
public:
    explicit BarrierSimulator(mlir::Operation* parentOp);

public:
    bool isDynamicBarriers() const {
        return _isDynamicBarriers;
    }

public:
    const BarrierConfig& getConfig(mlir::Value bar) const;

public:
    void checkProducerCount() const;
    void simulateBarriers();
    void linkNextIds();

private:
    void assignVirtualIds(mlir::Operation* parentOp);
    void parseBarriers(mlir::Operation* parentOp);
    void parseTasks(mlir::Operation* parentOp);
    static void cleanUpVirtualIds(mlir::Operation* parentOp);

private:
    enum class Status { Success, Skip, Fail };

    Status processSim(const VirtualDependencyTracker::Dependency& dep, BarrierUserConfig& user, int64_t count,
                      StringRef taskType, int64_t index, SmallVectorImpl<int64_t>& toVirtual,
                      RingBuffer<int64_t>& nextReal);

private:
    int64_t _availableBarriers = 0;
    int64_t _usedBarriers = 0;

    llvm::DenseMap<mlir::Operation*, size_t> _barriersMap;
    SmallVector<BarrierConfig> _barriers;
    bool _isDynamicBarriers = false;

    std::array<SmallVector<BarrierUserConfig>, MAX_DMA_ENGINES> _dmaTasks;
    SmallVector<BarrierUserConfig> _nceTasks;
    SmallVector<BarrierUserConfig> _actTasks;
    SmallVector<BarrierUserConfig> _upaTasks;

    VirtualDependencyTracker _vdt;

    Logger _log;
};

}  // namespace VPURT
}  // namespace vpux

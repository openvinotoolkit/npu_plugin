//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

#include <deque>
#include <iterator>
#include <sstream>
#include <string>

namespace vpux {

using SWTaskSignature = TaskSignature<VPUIP::SwKernelOp>;

template <class T>
unsigned countTasks(const SmallVector<std::pair<T, unsigned>>& vector) {
    return std::accumulate(vector.begin(), vector.end(), 0, [](const auto& a, const auto& b) {
        return a + b.second;
    });
}

// Create a string that should be placed as a suffix for operation name (Loc) with relevant metadata
// allowing post processing tools to correctly interpret profiling data
std::string createActShaveProfilingLocSuffix(size_t inDdrOffset, size_t clusterSize, size_t inClusterOffset,
                                             Optional<size_t> tileId);

// Gather profiling metadata from a profiled ActShave task
// Returns tuple: (size_t inDdrOffset, size_t clusterSize, size_t inClusterOffset and optional tile id)
std::tuple<size_t, size_t, size_t, Optional<size_t>> parseActShaveProfilingOffsets(mlir::Location loc);

// Update already existing profiling metadata which is a suffix task Loc setting with new
// offset. This is to be used when ActShave task with multiple SwKernelRun ops is being unrolled
mlir::Location getUpdatedActShaveProfilingLoc(mlir::Location loc, size_t inClusterOffset);

mlir::IntegerType getActShaveProfilingElementType(mlir::MLIRContext* ctx);

// Base pure virtual class for handling ActShave profiling
class BaseActShaveProfiler {
public:
    using ProfilingResults = SmallVector<std::pair<mlir::Value, unsigned>>;

private:
    // Return number of SwKernelRun tasks within this SwKernelOp
    static unsigned getNumProfiledTasks(VPUIP::SwKernelOp swOp) {
        auto swKernelRunIt = swOp.body().getOps<VPUIP::SwKernelRun>();
        return std::distance(swKernelRunIt.begin(), swKernelRunIt.end());
    }

public:
    BaseActShaveProfiler(unsigned clustersAmount, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                         vpux::IndexedSymbolAttr memKindAttr, mlir::FuncOp netFunc, vpux::Logger& log,
                         std::shared_ptr<NameUniqifier> uniqifier);

    // Get amount of memory needed to store profiling data of all ActShave tasks in the model
    unsigned getRequiredDdrMemory() const;

    // Go over all SwKernelOps and store required information about those tasks like required size of
    // profiling buffer or size of profiling buffer instances
    void scheduleTask(VPUIP::SwKernelOp swOp);

    // Main function which goes through all identified ActShave ops and based on gathered data recreates
    // those operations to have profiling output with proper slot in profiling buffer instance. When profiling
    // buffer is full it also inserts CMX2DDR DMA and allocates new profiling buffer
    void addProfilingOps(mlir::BlockArgument& profilingDdrResult, SmallVector<mlir::Value>& clusterResults);

protected:
    // Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
    // new one needs to be allocated
    virtual mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) = 0;

    // Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
    // profiling buffer instance is full or there are no more tasks to profile
    virtual mlir::Value copyToDdr(ProfilingResults profilingResults, mlir::Operation* cmxMemOp,
                                  size_t& currentDDROffset, mlir::BlockArgument& profilingDdrResult) = 0;

    // Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
    virtual mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                        unsigned numTasks) = 0;

    // Replace a Actshave task with new one that has profiling output set
    virtual mlir::Value replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask, mlir::Value profilingBuffer,
                                                mlir::Location loc) = 0;

    SWTaskSignature getTaskSignature(VPUIP::SwKernelOp swOp) const;

    mlir::Type getTimestampType(unsigned tasksAmount);

protected:
    unsigned _clustersAmount;
    unsigned _profilingWorkloadSize;
    unsigned _profilingElementSize;
    std::deque<unsigned> _profilingBufferSizes;
    SmallVector<SWTaskSignature> _swTaskSignatures;
    mlir::OpBuilder& _builder;
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _netFunc;
    vpux::IndexedSymbolAttr _memKindAttr;
    vpux::Logger& _log;
    std::shared_ptr<NameUniqifier> _uniqifier;
};

// Class for handling ActShave profiling if none of Actshave tasks in the model is multiclustered.
// This way profiling buffer instance can be a simple memref
class UniformNonTiledActShaveProfiler : public BaseActShaveProfiler {
public:
    UniformNonTiledActShaveProfiler(unsigned clustersAmount, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                                    vpux::IndexedSymbolAttr memKindAttr, mlir::FuncOp netFunc, vpux::Logger& log,
                                    std::shared_ptr<NameUniqifier> uniqifier);

protected:
    // Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
    // new one needs to be allocated. Type of this alloc is a memref
    mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) override;

    // Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
    // profiling buffer instance is full or there are no more tasks to profile
    mlir::Value copyToDdr(ProfilingResults profilingResults, mlir::Operation* cmxMemOp, size_t& currentDDROffset,
                          mlir::BlockArgument& profilingDdrResult) override;

    // Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
    mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                unsigned numTasks) override;

    // Replace a Actshave task with new one that has profiling output set
    mlir::Value replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask, mlir::Value profilingBuffer,
                                        mlir::Location loc) override;
};

// Class for handling ActShave profiling if at least one of ActShave tasks in the model is multiclustered.
// Profiling buffer will be represented as a DistributedBuffer
class NCETiledActShaveProfiler : public BaseActShaveProfiler {
private:
    VPUIP::DistributedBufferType getDistributedBufferType(unsigned totalElements);

public:
    NCETiledActShaveProfiler(unsigned clustersAmount, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                             vpux::IndexedSymbolAttr memKindAttr, mlir::FuncOp netFunc, vpux::Logger& log,
                             std::shared_ptr<NameUniqifier> uniqifier);

protected:
    // Create allocation operation representing profiling buffer instance in CMX. If such buffer is full
    // new one needs to be allocated. Type of this alloc is a DistributedBufferType
    virtual mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location) override;

    // Insert DMA that will copy profiling buffer instance to proper offset in profiling output once
    // profiling buffer instance is full or there are no more tasks to profile
    virtual mlir::Value copyToDdr(ProfilingResults profilingResults, mlir::Operation* cmxMemOp,
                                  size_t& currentDDROffset, mlir::BlockArgument& profilingDdrResult) override;

    // Get a SubView of profiling buffer instance so that given ActShave task is given required chunk of it
    virtual mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned profilingSamplesInCMX,
                                        unsigned numTasks) override;

    // Replace a Actshave task with new one that has profiling output set. If this task is not multiclustered
    // then additional cast (ViewOp) is inserted for profiling slot to maintain type compatibility
    virtual mlir::Value replaceOpWithProfiledOp(VPUIP::SwKernelOp origSwTask, mlir::Value profilingBuffer,
                                                mlir::Location loc) override;
};

}  // namespace vpux

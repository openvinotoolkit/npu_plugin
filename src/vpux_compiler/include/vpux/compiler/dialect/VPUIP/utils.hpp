//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// Profiling
//

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR_30XX = 0x208200BC;
constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR_37XX = 0x26029000;
// DMA Profiling consist of 2 32bit timestamps
constexpr uint16_t HW_DMA_PROFILING_SIZE_BYTES = 8;
constexpr uint32_t HW_DMA_PROFILING_MAX_BUFFER_SIZE = 256;
constexpr uint32_t HW_DMA_PROFILING_RESERVED_MEM_OFFSET = 0;
// DPU Profiling consist of 2 64bit timestamps(start and stop)
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_30XX = 16;
// DPU Profiling for 37XX use MODE0: // 8’h0, odu_tstamp[27:0], odu_wl_duration[27:0], {3’h0,sve_id[4:0]},
// idu_tstamp[27:0], idu_wl_duration[27:0]
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_37XX = 16;
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_40XX = 16;
constexpr uint32_t HW_DPU_PROFILING_MAX_BUFFER_SIZE =
        1024;  // Up to 64 DPU Tasks in single CMX DPU profiling buffer instance
// UPA Profiling consist of 2 64bit timestamps(start and stop) + 2 32bit for active and stall counters
constexpr uint16_t HW_UPA_PROFILING_SIZE_BYTES = 24;
// ActShave Profiling consist of 1 64bit start timestamp + 1 32bit duration + 1 32bit stall counter
constexpr uint16_t HW_ACT_SHAVE_PROFILING_SIZE_BYTES = 16;
// ActShave Profiling buffer size in bytes
constexpr uint32_t HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE = 128;
// 3720 architecture has 2 Mb CMX size. This define describes its half.
constexpr uint32_t HW_MIDDLE_OF_AVAILABLE_CMX_MEMORY_3720 = (1024 * 1024);

uint16_t getProfWorkloadSize(mlir::ModuleOp module);

//
// Run-time info
//

double getMemoryDerateFactor(IE::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(IE::MemoryResourceOp mem);
int64_t getNumClusterUsed(mlir::ModuleOp module);
int64_t getNumAvailableBarriers(mlir::Operation* parentOp);

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);
mlir::Value getTopBufferOfNCEClusterTiling(mlir::Operation* innerOp, mlir::Value buffer);

// Sparsity utils for optimize-copies pass family
void moveRootAllocBefore(mlir::Operation* root, mlir::Operation* targerOp);
mlir::Type extractDataType(mlir::Type type);
mlir::Type extractDataType(mlir::Value value);

// Return operation which allocate memory buffer. Note, that
// For sparse data rootAlloc look like this:
// val <=== VPUIP.GroupSparseBuffer <-- AllocatorOp
//                                 \<-- [AllocatorOp] # optional sparsity map
template <class AllocatorOp, typename = std::enable_if<std::is_same<AllocatorOp, mlir::memref::AllocOp>::value ||
                                                       std::is_same<AllocatorOp, VPURT::AllocDistributed>::value>>
mlir::Operation* getRootAlloc(mlir::Value val) {
    if (auto rootGroup = val.getDefiningOp<VPUIP::GroupSparseBufferOp>()) {
        if (rootGroup.data().getDefiningOp<AllocatorOp>() == nullptr) {
            return nullptr;
        }
        // TODO: Handle SET
        const auto sparsityMap = rootGroup.sparsityMap();
        if (sparsityMap && sparsityMap.getDefiningOp<AllocatorOp>() == nullptr) {
            return nullptr;
        }
        return rootGroup;
    }
    return val.getDefiningOp<AllocatorOp>();
}

mlir::Operation* getRootConst(mlir::Value val);

//
// Unrolling Utilities
//

SmallVector<mlir::Value> getPerClusterBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                              mlir::Value clusterOperand, mlir::Value innerOperand, int64_t numClusters,
                                              mlir::PatternRewriter& rewriter, bool allowDiscontinuousBuffers = false);
SmallVector<mlir::Value> getSplitBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                         mlir::Value operand, SmallVector<vpux::Shape> shapes,
                                         SmallVector<vpux::Shape> shapeOffsets, int64_t splitNum,
                                         mlir::PatternRewriter& rewriter);

//
// MovePureViewOpBeforeCopy Utilities
//

bool isSegmentedOverH(VPU::DistributedTensorAttr distAttr);
bool isSegmentedOverC(VPU::DistributedTensorAttr distAttr);
VPU::DistributedTensorAttr getSOHDistAttrWithNewShape(mlir::MLIRContext* ctx, VPUIP::DistributedBufferType origDistType,
                                                      ShapeRef newShape);
bool isDistributedCompatibleAfterShapeChange(VPUIP::DistributedBufferType inDistType, ShapeRef shape);

//
// Distributed buffer type compatibility check
//

bool equalPerClusterShapes(VPUIP::DistributedBufferType distributedBufferType);
bool isCompatibleForDistributedInputOutput(mlir::Operation* op, VPUIP::DistributedBufferType distributedInType,
                                           VPUIP::DistributedBufferType distributedOutType);
int64_t getTilingDimIndex(VPUIP::DistributedBufferType distributedBufferType);
bool isMemoryContiguousWithTiling(VPUIP::DistributedBufferType distributedBufferType);

}  // namespace VPUIP
}  // namespace vpux

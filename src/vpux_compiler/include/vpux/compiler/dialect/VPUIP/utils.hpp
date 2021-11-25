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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"

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

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR = 0x208200BC;
// DMA Profiling consist of 2 32bit timestamps
constexpr uint16_t HW_DMA_PROFILING_SIZE_BYTES = 8;
constexpr uint32_t HW_DMA_PROFILING_MAX_BUFFER_SIZE = 256;
// DPU Profiling consist of 2 64bit timestamps(start and stop)
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES = 16;
constexpr uint32_t HW_DPU_PROFILING_MAX_BUFFER_SIZE = 128;
// UPA Profiling consist of 2 64bit timestamps(start and stop) + 2 32bit for active and stall counters
constexpr uint16_t HW_UPA_PROFILING_SIZE_BYTES = 24;

//
// Run-time info
//

double getMemoryDerateFactor(IERT::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(IERT::MemoryResourceOp mem);
double getProcessorFrequency(IERT::ExecutorResourceOp res);

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

//
// Serialize utils
//

MVCNN::TargetDevice mapTargetDevice(VPU::ArchKind kind);
MVCNN::TargetDeviceRevision mapTargetDeviceRevision(VPU::ArchKind kind);

MVCNN::DType createDType(mlir::Type type);
MVCNN::MemoryLocation createMemoryLocation(VPURT::BufferSection section);
MVCNN::order3 createOrder3(mlir::ArrayAttr attr);

const EnumMap<vpux::PreProcessColorSpace, MVCNN::PreProcessColorSpace> mapPreProcessColorFormat = {
        {vpux::PreProcessColorSpace::BGR, MVCNN::PreProcessColorSpace::PreProcessColorSpace_BGR},
        {vpux::PreProcessColorSpace::RGB, MVCNN::PreProcessColorSpace::PreProcessColorSpace_RGB},
        {vpux::PreProcessColorSpace::NV12, MVCNN::PreProcessColorSpace::PreProcessColorSpace_NV12},
        {vpux::PreProcessColorSpace::I420, MVCNN::PreProcessColorSpace::PreProcessColorSpace_I420},
        {vpux::PreProcessColorSpace::NONE, MVCNN::PreProcessColorSpace::PreProcessColorSpace_DEFAULT},
};

const EnumMap<vpux::PreProcessResizeAlgorithm, MVCNN::PreProcessResizeAlgorithm> mapPreProcessResizeAlgorithm = {
        {vpux::PreProcessResizeAlgorithm::RESIZE_BILINEAR,
         MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_BILINEAR},
        {vpux::PreProcessResizeAlgorithm::RESIZE_AREA,
         MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_AREA},
        {vpux::PreProcessResizeAlgorithm::NO_RESIZE,
         MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_NO_RESIZE},
};

}  // namespace VPUIP
}  // namespace vpux

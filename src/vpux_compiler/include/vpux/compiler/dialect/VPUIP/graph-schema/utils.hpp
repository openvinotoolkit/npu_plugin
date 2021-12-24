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
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// Serialize utils
//

MVCNN::TargetDevice mapTargetDevice(VPU::ArchKind kind);
MVCNN::TargetDeviceRevision mapTargetDeviceRevision(VPU::ArchKind kind);

MVCNN::DType createDType(mlir::Type type);
MVCNN::MemoryLocation createMemoryLocation(VPURT::BufferSection section);
MVCNN::order3 createOrder3(mlir::ArrayAttr attr);

extern const EnumMap<vpux::PreProcessColorSpace, MVCNN::PreProcessColorSpace> mapPreProcessColorFormat;
extern const EnumMap<vpux::PreProcessResizeAlgorithm, MVCNN::PreProcessResizeAlgorithm> mapPreProcessResizeAlgorithm;

}  // namespace VPUIP
}  // namespace vpux

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <openvino/core/type/element_type.hpp>

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

extern const EnumMap<ov::element::Type_t, MVCNN::OVNodeType> mapElementType;
extern const EnumMap<vpux::PreProcessColorSpace, MVCNN::PreProcessColorSpace> mapPreProcessColorFormat;
extern const EnumMap<vpux::PreProcessResizeAlgorithm, MVCNN::PreProcessResizeAlgorithm> mapPreProcessResizeAlgorithm;

}  // namespace VPUIP
}  // namespace vpux

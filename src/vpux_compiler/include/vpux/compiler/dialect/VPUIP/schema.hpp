//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <flatbuffers/flatbuffers.h>

#include <vpux/compiler/dialect/VPUIP/generated/schema/graphfile_generated.h>

//
// stringifyEnum
//

namespace MVCNN {

#define VPUX_STRINGIFY_SCHEMA_ENUM(_name_)             \
    inline vpux::StringRef stringifyEnum(_name_ val) { \
        return VPUX_COMBINE(EnumName, _name_)(val);    \
    }

VPUX_STRINGIFY_SCHEMA_ENUM(PhysicalProcessor)
VPUX_STRINGIFY_SCHEMA_ENUM(DMAEngine)
VPUX_STRINGIFY_SCHEMA_ENUM(PhysicalMem)
VPUX_STRINGIFY_SCHEMA_ENUM(ExecutionFlag)
VPUX_STRINGIFY_SCHEMA_ENUM(SpecificTask)
VPUX_STRINGIFY_SCHEMA_ENUM(MemoryLocation)
VPUX_STRINGIFY_SCHEMA_ENUM(DType)
VPUX_STRINGIFY_SCHEMA_ENUM(ControllerSubTask)
VPUX_STRINGIFY_SCHEMA_ENUM(DataType)
VPUX_STRINGIFY_SCHEMA_ENUM(PSROIPoolingMode)
VPUX_STRINGIFY_SCHEMA_ENUM(InterpolationMethod)
VPUX_STRINGIFY_SCHEMA_ENUM(EltwiseParam)
VPUX_STRINGIFY_SCHEMA_ENUM(EltwisePostOpsNestedParams)
VPUX_STRINGIFY_SCHEMA_ENUM(PostOpsNestedParams)
VPUX_STRINGIFY_SCHEMA_ENUM(UnaryOpNestedParams)
VPUX_STRINGIFY_SCHEMA_ENUM(SoftwareLayerParams)
VPUX_STRINGIFY_SCHEMA_ENUM(NN2Optimization)
VPUX_STRINGIFY_SCHEMA_ENUM(DPULayerType)
VPUX_STRINGIFY_SCHEMA_ENUM(PPELayerType)
VPUX_STRINGIFY_SCHEMA_ENUM(MPE_Mode)
VPUX_STRINGIFY_SCHEMA_ENUM(PPERoundingMode)

#undef VPUX_STRINGIFY_SCHEMA_ENUM

}  // namespace MVCNN

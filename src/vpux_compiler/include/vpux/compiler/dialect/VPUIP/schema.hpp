//
// Copyright 2020 Intel Corporation.
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

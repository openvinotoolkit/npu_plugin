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

#include "vpux/utils/core/string_ref.hpp"

#include <flatbuffers/flatbuffers.h>

#include <vpux/compiler/dialect/VPUIP/generated/schema/graphfile_generated.h>

//
// stringifyEnum
//

namespace MVCNN {

inline vpux::StringRef stringifyEnum(PSROIPoolingMode val) {
    return EnumNamePSROIPoolingMode(val);
}

inline vpux::StringRef stringifyEnum(DataType val) {
    return EnumNameDataType(val);
}

inline vpux::StringRef stringifyEnum(InterpolationMethod val) {
    return EnumNameInterpolationMethod(val);
}

inline vpux::StringRef stringifyEnum(SoftwareLayer val) {
    return EnumNameSoftwareLayer(val);
}

inline vpux::StringRef stringifyEnum(NNHelper val) {
    return EnumNameNNHelper(val);
}

inline vpux::StringRef stringifyEnum(EltwiseParam val) {
    return EnumNameEltwiseParam(val);
}

inline vpux::StringRef stringifyEnum(EltwisePostOpsNestedParams val) {
    return EnumNameEltwisePostOpsNestedParams(val);
}

inline vpux::StringRef stringifyEnum(PostOpsNestedParams val) {
    return EnumNamePostOpsNestedParams(val);
}

inline vpux::StringRef stringifyEnum(UnaryOpNestedParams val) {
    return EnumNameUnaryOpNestedParams(val);
}

inline vpux::StringRef stringifyEnum(SoftwareLayerParams val) {
    return EnumNameSoftwareLayerParams(val);
}

inline vpux::StringRef stringifyEnum(NN2Optimization val) {
    return EnumNameNN2Optimization(val);
}

inline vpux::StringRef stringifyEnum(DPULayerType val) {
    return EnumNameDPULayerType(val);
}

inline vpux::StringRef stringifyEnum(PPELayerType val) {
    return EnumNamePPELayerType(val);
}

inline vpux::StringRef stringifyEnum(MPE_Mode val) {
    return EnumNameMPE_Mode(val);
}

inline vpux::StringRef stringifyEnum(ControllerSubTask val) {
    return EnumNameControllerSubTask(val);
}

inline vpux::StringRef stringifyEnum(MemoryLocation val) {
    return EnumNameMemoryLocation(val);
}

inline vpux::StringRef stringifyEnum(DType val) {
    return EnumNameDType(val);
}

inline vpux::StringRef stringifyEnum(ExecutionFlag val) {
    return EnumNameExecutionFlag(val);
}

inline vpux::StringRef stringifyEnum(SpecificTask val) {
    return EnumNameSpecificTask(val);
}

}  // namespace MVCNN

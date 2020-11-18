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

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <flatbuffers/flatbuffers.h>

#include <vpux/utils/VPUIP/schema/generated/graphfile_generated.h>

//
// EnumTraits
//

namespace vpux {

template <>
struct EnumTraits<MVCNN::PSROIPoolingMode> {
    static StringRef getEnumValueName(MVCNN::PSROIPoolingMode val) {
        return MVCNN::EnumNamePSROIPoolingMode(val);
    }
};

template <>
struct EnumTraits<MVCNN::DataType> {
    static StringRef getEnumValueName(MVCNN::DataType val) {
        return MVCNN::EnumNameDataType(val);
    }
};

template <>
struct EnumTraits<MVCNN::InterpolationMethod> {
    static StringRef getEnumValueName(MVCNN::InterpolationMethod val) {
        return MVCNN::EnumNameInterpolationMethod(val);
    }
};

template <>
struct EnumTraits<MVCNN::SoftwareLayer> {
    static StringRef getEnumValueName(MVCNN::SoftwareLayer val) {
        return MVCNN::EnumNameSoftwareLayer(val);
    }
};

template <>
struct EnumTraits<MVCNN::NNHelper> {
    static StringRef getEnumValueName(MVCNN::NNHelper val) {
        return MVCNN::EnumNameNNHelper(val);
    }
};

template <>
struct EnumTraits<MVCNN::EltwiseParam> {
    static StringRef getEnumValueName(MVCNN::EltwiseParam val) {
        return MVCNN::EnumNameEltwiseParam(val);
    }
};

template <>
struct EnumTraits<MVCNN::EltwisePostOpsNestedParams> {
    static StringRef getEnumValueName(MVCNN::EltwisePostOpsNestedParams val) {
        return MVCNN::EnumNameEltwisePostOpsNestedParams(val);
    }
};

template <>
struct EnumTraits<MVCNN::PostOpsNestedParams> {
    static StringRef getEnumValueName(MVCNN::PostOpsNestedParams val) {
        return MVCNN::EnumNamePostOpsNestedParams(val);
    }
};

template <>
struct EnumTraits<MVCNN::UnaryOpNestedParams> {
    static StringRef getEnumValueName(MVCNN::UnaryOpNestedParams val) {
        return MVCNN::EnumNameUnaryOpNestedParams(val);
    }
};

template <>
struct EnumTraits<MVCNN::SoftwareLayerParams> {
    static StringRef getEnumValueName(MVCNN::SoftwareLayerParams val) {
        return MVCNN::EnumNameSoftwareLayerParams(val);
    }
};

template <>
struct EnumTraits<MVCNN::NN2Optimization> {
    static StringRef getEnumValueName(MVCNN::NN2Optimization val) {
        return MVCNN::EnumNameNN2Optimization(val);
    }
};

template <>
struct EnumTraits<MVCNN::DPULayerType> {
    static StringRef getEnumValueName(MVCNN::DPULayerType val) {
        return MVCNN::EnumNameDPULayerType(val);
    }
};

template <>
struct EnumTraits<MVCNN::PPELayerType> {
    static StringRef getEnumValueName(MVCNN::PPELayerType val) {
        return MVCNN::EnumNamePPELayerType(val);
    }
};

template <>
struct EnumTraits<MVCNN::MPE_Mode> {
    static StringRef getEnumValueName(MVCNN::MPE_Mode val) {
        return MVCNN::EnumNameMPE_Mode(val);
    }
};

template <>
struct EnumTraits<MVCNN::ControllerSubTask> {
    static StringRef getEnumValueName(MVCNN::ControllerSubTask val) {
        return MVCNN::EnumNameControllerSubTask(val);
    }
};

template <>
struct EnumTraits<MVCNN::MemoryLocation> {
    static StringRef getEnumValueName(MVCNN::MemoryLocation val) {
        return MVCNN::EnumNameMemoryLocation(val);
    }
};

template <>
struct EnumTraits<MVCNN::DType> {
    static StringRef getEnumValueName(MVCNN::DType val) {
        return MVCNN::EnumNameDType(val);
    }
};

template <>
struct EnumTraits<MVCNN::ExecutionFlag> {
    static StringRef getEnumValueName(MVCNN::ExecutionFlag val) {
        return MVCNN::EnumNameExecutionFlag(val);
    }
};

template <>
struct EnumTraits<MVCNN::SpecificTask> {
    static StringRef getEnumValueName(MVCNN::SpecificTask val) {
        return MVCNN::EnumNameSpecificTask(val);
    }
};

}  // namespace vpux

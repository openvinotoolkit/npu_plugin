//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/m2i_utils.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::M2iColorFmt VPU::IEtoM2iColorFmt(IE::ColorFmt fmt) {
    if (fmt == IE::ColorFmt::NV12) {  // semi-planar
        return VPU::M2iColorFmt::SP_NV12_8;
    } else if (fmt == IE::ColorFmt::I420) {  // planar
        return VPU::M2iColorFmt::PL_YUV420_8;
    } else if (fmt == IE::ColorFmt::RGB) {  // C-minor
        return VPU::M2iColorFmt::IL_RGB888;
    } else if (fmt == IE::ColorFmt::BGR) {  // C-minor
        return VPU::M2iColorFmt::IL_BGR888;
    } else {
        VPUX_THROW("IEtoM2iColorFmt: unsupported format {0}", fmt);
    }
}

long VPU::getM2iLineStride(NDTypeInterface ndType, size_t dimW) {
    auto shape = ndType.getShape().raw();
    auto lineStride = static_cast<long>(shape[dimW] * (ndType.getElemTypeSize().count() / CHAR_BIT));
    return lineStride;
}

// Line stride must be multiple of 16
bool VPU::isM2iLineStrideSupported(long lineStride) {
    return lineStride % 16 == 0;
}

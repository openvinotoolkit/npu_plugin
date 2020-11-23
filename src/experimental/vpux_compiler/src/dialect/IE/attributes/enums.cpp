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

#include "vpux/compiler/dialect/IE/attributes/enums.hpp"

using namespace vpux;

//
// Layout utilities
//

int32_t vpux::IE::getRank(Layout layout) {
    switch (layout) {
    case IE::Layout::C:
        return 1;

    case IE::Layout::NC:
        return 2;

    case IE::Layout::CHW:
        return 3;

    case IE::Layout::NCHW:
    case IE::Layout::NHWC:
        return 4;

    case IE::Layout::NCDHW:
    case IE::Layout::NDHWC:
        return 5;

    default:
        return 0;
    }
}

DimsOrder vpux::IE::getDimsOrder(Layout layout) {
    switch (layout) {
    case IE::Layout::SCALAR:
        return DimsOrder::fromNumDims(0);
    case IE::Layout::C:
        return DimsOrder::C;
    case IE::Layout::NC:
        return DimsOrder::NC;
    case IE::Layout::CHW:
        return DimsOrder::CHW;
    case IE::Layout::NCHW:
        return DimsOrder::NCHW;
    case IE::Layout::NHWC:
        return DimsOrder::NHWC;
    case IE::Layout::NCDHW:
        return DimsOrder::NCDHW;
    case IE::Layout::NDHWC:
        return DimsOrder::NDHWC;
    default:
        VPUX_THROW("Can't convert IE::Layout '{0}' to DimsOrder", layout);
    }
}

mlir::AffineMap vpux::IE::getAffineMap(mlir::MLIRContext* ctx, Layout layout) {
    return getDimsOrder(layout).toAffineMap(ctx);
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/enums.cpp.inc>

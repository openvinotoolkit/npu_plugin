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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/compiler/act_kernels/act_kernel_gen.h"

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/movitools/movitools.h"

using namespace vpux;
using namespace mlir;

namespace vpux {
namespace VPUIP {

VPUIP::BlobWriter::SpecificTask ACTShaveTaskOp::serialize(VPUIP::BlobWriter& writer) {
    return writer.createACTShaveTask(*this);
}

VPUIP::BlobWriter::SpecificTask  SW_Kernel::serialize(vpux::VPUIP::BlobWriter&) {
    return {};
}

}  // namespace VPUIP
}  // namespace vpux
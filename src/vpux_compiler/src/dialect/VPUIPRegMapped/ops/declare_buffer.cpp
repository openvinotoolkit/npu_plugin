//
// Copyright (C) 2022 Intel Corporation.
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

#include <host_parsed_inference.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {
// Round up val by N
template <size_t N>
size_t roundUp(size_t t) {
    return static_cast<size_t>((t + N - 1) & ~(N - 1));
}
}  // namespace

void VPUIPRegMapped::DeclareBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);

    // no serialization for declareBuffer

    return;
}

size_t VPUIPRegMapped::DeclareBufferOp::getBinarySize() {
    constexpr uint8_t BITS_IN_BYTE = 8;
    return roundUp<BITS_IN_BYTE>(getType().getSizeInBits() / BITS_IN_BYTE);
}

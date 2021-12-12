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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <host_parsed_inference.h>

using namespace vpux;

namespace {
// Round up val by N
template <size_t N>
uint32_t round_up(uint32_t t) {
    return static_cast<uint32_t>((t + N - 1) & ~(N - 1));
}
}

void VPUIPRegMapped::DeclareBufferOp::serialize(std::vector<char>& buff) {
    //no serialization for declareBuffer
    return;
}

size_t VPUIPRegMapped::DeclareBufferOp::getBinarySize() {
    constexpr uint8_t BITS_IN_BYTE = 8;
    return round_up<BITS_IN_BYTE>(getType().getSizeInBits() / BITS_IN_BYTE);
}

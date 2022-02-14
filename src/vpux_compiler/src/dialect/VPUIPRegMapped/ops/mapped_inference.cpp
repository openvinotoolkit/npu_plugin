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

//
// MappedInferenceOp
//

void vpux::VPUIPRegMapped::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    host_parsing::MappedInference mi;
    memset(reinterpret_cast<void*>(&mi), 0, getBinarySize());

    mi.dmaTasks[0].count = dmaCount();
    mi.invariants.count = invariantCount();
    mi.variants.count = variantCount();
    mi.actKInvocations.count = actInvocationsCount();
    mi.barrierConfigs.count = barrierCount();

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&mi);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::MappedInferenceOp::getBinarySize() {
    return sizeof(host_parsing::MappedInference);
}

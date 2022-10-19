//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/mem_size.hpp"

using namespace vpux;

namespace vpux {
namespace VPUIPRegMapped {

// structure containing the Activation Shave Configs that the runtime expects
// better implementation TBD
struct ActShaveRtConfigs {
    static constexpr uint32_t stackFrame0Value = 0x2E000800;
    static constexpr uint32_t stackFrame1Value = 0x2E000C00;
    static constexpr uint32_t stackFrame2Value = 0x2E200800;
    static constexpr uint32_t stackFrame3Value = 0x2E200C00;
    static constexpr vpux::KB stackSize = 16_KB;
    static constexpr bool useScheduleEmbeddedRt = false;
    static constexpr vpux::MB codeWindowBufferSize = 1_MB;
};

constexpr vpux::KB ActShaveRtConfigs::stackSize;
constexpr vpux::MB ActShaveRtConfigs::codeWindowBufferSize;

}  // namespace VPUIPRegMapped
}  // namespace vpux

//
// MappedInferenceOp
//

void vpux::VPUIPRegMapped::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    host_parsing::MappedInference mi;
    memset(reinterpret_cast<void*>(&mi), 0, getBinarySize());

    mi.dmaTasks[0].count = dmaCount();
    mi.invariants.count = invariantCount();
    mi.variants.count = variantCount();
    mi.actKRanges.count = actKernelRangesCount();
    mi.actKInvocations.count = actKernelInvocationsCount();
    mi.barrierConfigs.count = barrierCount();

    if (mi.actKInvocations.count) {
        mi.actRtConfigs.stackFrames_[0] = vpux::VPUIPRegMapped::ActShaveRtConfigs::stackFrame0Value;
        mi.actRtConfigs.stackFrames_[1] = vpux::VPUIPRegMapped::ActShaveRtConfigs::stackFrame1Value;
        mi.actRtConfigs.stackFrames_[2] = vpux::VPUIPRegMapped::ActShaveRtConfigs::stackFrame2Value;
        mi.actRtConfigs.stackFrames_[3] = vpux::VPUIPRegMapped::ActShaveRtConfigs::stackFrame3Value;

        mi.actRtConfigs.stackSize_ = vpux::VPUIPRegMapped::ActShaveRtConfigs::stackSize.to<vpux::Byte>().count();
        mi.actRtConfigs.useScheduleEmbeddedRt_ = vpux::VPUIPRegMapped::ActShaveRtConfigs::useScheduleEmbeddedRt;
        mi.actRtConfigs.codeWindowBufferSize_ =
                vpux::VPUIPRegMapped::ActShaveRtConfigs::codeWindowBufferSize.to<vpux::Byte>().count();
        mi.actRtConfigs.actRtWindowBase_ = 0;
    }

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&mi);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::MappedInferenceOp::getBinarySize() {
    return sizeof(host_parsing::MappedInference);
}

mlir::FailureOr<uint64_t> vpux::VPUIPRegMapped::MappedInferenceOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == this->dmaTasks()) {
        return offsetof(host_parsing::MappedInference, dmaTasks);
    } else if (val == this->barrierTasks()) {
        return offsetof(host_parsing::MappedInference, barrierConfigs);
    }

    return mlir::failure();
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/Support/FileSystem.h>
#include <mlir/IR/BuiltinTypes.h>
#include <fstream>
#include <vector>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/scope_exit.hpp"

using namespace vpux;

//
// DeclareKernelArgsOp
//

void vpux::VPUIPRegMapped::DeclareKernelArgsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto secDataSizePair = vpux::ELF::getDataAndSizeOfElfSection(elfBlob, {".data", ".arg.data"});

    binDataSection.appendData(secDataSizePair.first, secDataSizePair.second);
}

size_t vpux::VPUIPRegMapped::DeclareKernelArgsOp::getBinarySize() {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto secDataSizePair = vpux::ELF::getDataAndSizeOfElfSection(elfBlob, {".data", ".arg.data"});

    return secDataSizePair.second;
}

// The .data sections for the sw layers must be 1kB aligned as an ActShave requirement
size_t vpux::VPUIPRegMapped::DeclareKernelArgsOp::getAlignmentRequirements() {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::DeclareKernelArgsOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::SW_KernelText;
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::DeclareKernelArgsOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::DeclareKernelArgsOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

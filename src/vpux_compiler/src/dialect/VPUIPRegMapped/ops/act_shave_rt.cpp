//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/Support/FileSystem.h>
#include <mlir/IR/BuiltinTypes.h>
#include <fstream>
#include <vector>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/scope_exit.hpp"

using namespace vpux;

//
// ActShaveRtOp
//

void vpux::VPUIPRegMapped::ActShaveRtOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto secDataSizePair = vpux::ELF::getDataAndSizeOfElfSection(elfBlob, {".text"});

    binDataSection.appendData(secDataSizePair.first, secDataSizePair.second);
}

size_t vpux::VPUIPRegMapped::ActShaveRtOp::getBinarySize() {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto secDataSizePair = vpux::ELF::getDataAndSizeOfElfSection(elfBlob, {".text"});

    return secDataSizePair.second;
}

uint32_t vpux::VPUIPRegMapped::ActShaveRtOp::getKernelEntry() {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto accessor = elf::ElfDDRAccessManager(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}

uint32_t vpux::VPUIPRegMapped::ActShaveRtOp::getVersion() {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto secDataSizePair = vpux::ELF::getDataAndSizeOfElfSection(elfBlob, {".versiondata"});

    auto nnActEntryRtVersion = reinterpret_cast<const uint32_t*>(secDataSizePair.first);

    return *nnActEntryRtVersion;
}

// The management kernel code must be 1kB aligned as an ActShave requirement
size_t vpux::VPUIPRegMapped::ActShaveRtOp::getAlignmentRequirements() {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::ActShaveRtOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::SW_KernelText;
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ActShaveRtOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ActShaveRtOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

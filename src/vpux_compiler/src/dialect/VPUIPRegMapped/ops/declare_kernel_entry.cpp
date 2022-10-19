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
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

//
// DeclareKernelEntryOp
//

uint32_t vpux::VPUIPRegMapped::DeclareKernelEntryOp::getKernelEntry() {
    auto kernel = kernel_path();

    SmallString currPath;

    if (std::getenv("KMB_PLUGIN_HOME")) {
        currPath.append(std::getenv("KMB_PLUGIN_HOME"));
    }

    currPath.append(sw_layers_elf_path);
    llvm::sys::fs::set_current_path(currPath);

    std::ifstream stream(kernel.str(), std::ios::binary);
    VPUX_THROW_WHEN(stream.fail(), "Cannot access elf file at path {0}{1}", currPath, kernel.str());

    std::vector<char> elfBlob((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    stream.close();

    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>((const uint8_t*)elfBlob.data(), elfBlob.size());

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}

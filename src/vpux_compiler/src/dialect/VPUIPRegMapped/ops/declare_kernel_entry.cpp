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

#include <host_parsed_inference.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include <vpux_elf/reader32.hpp>
#include <fstream>
#include <vector>
#include <iostream>

using namespace vpux;

//
// DeclareKernelArgsOp
//

uint32_t vpux::VPUIPRegMapped::DeclareKernelEntryOp::getKernelEntry() {

    auto kernel = kernel_path();

    // auto base_kernels_dir = "/sw_runtime_kernels/kernels/prebuild/";

    // auto kernel_path = base_kernels_dir + kernel;

    std::ifstream stream(kernel.str(), std::ios::binary);
    std::vector<char> elfBlob((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    stream.close();

    elf::Reader32 elf_reader;

    elf_reader.loadElf(elfBlob.data(), elfBlob.size());

    const elf::ELF32Header* actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;

}

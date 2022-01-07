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

void vpux::VPUIPRegMapped::DeclareKernelArgsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {

    auto kernel = kernel_path();

    // auto base_kernels_dir = "/sw_runtime_kernels/kernels/prebuild/";

    // auto full_kernel_path = base_kernels_dir + kernel;

    std::ifstream stream(kernel.str(), std::ios::binary);
    std::vector<char> elfBlob((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    stream.close();

    elf::Reader32 elf_reader;

    elf_reader.loadElf(elfBlob.data(), elfBlob.size());

    uint8_t* sec_data;
    uint32_t sec_size;

    for (size_t i = 0; i < elf_reader.getSectionsNum(); ++i) {
        auto section = elf_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();

        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0){ //just data perhaps
            sec_size = sectionHeader->sh_size;
            sec_data = (uint8_t*)section.getData<uint8_t>();
            break;
        } 
    }

    binDataSection.appendData((const uint8_t*)sec_data, sec_size);
}

size_t vpux::VPUIPRegMapped::DeclareKernelArgsOp::getBinarySize() {

    auto kernel_path = "/home/pcarabas/work/shave_kernels/hswish/build/3010xx/kops/hswish_fp16_kops.elf";

    std::ifstream stream(kernel_path, std::ios::binary);
    std::vector<char> elfBlob((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    stream.close();

    elf::Reader32 elf_reader;

    elf_reader.loadElf(elfBlob.data(), elfBlob.size());

    for (size_t i = 0; i < elf_reader.getSectionsNum(); ++i) {
        auto section = elf_reader.getSection(i);
        const auto sec_name = section.getName();
        const auto sectionHeader = section.getHeader();

        if (strcmp(sec_name, ".data") == 0 || strcmp(sec_name, ".arg.data") == 0){ //just data perhaps
            return sectionHeader->sh_size;
        } 
    }

    return -1;
}

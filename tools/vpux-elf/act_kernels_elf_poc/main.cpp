//
// Copyright Intel Corporation.
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

#include <vpux/compiler/dialect/VPUIP/elf_blob_serializer.hpp>

#include <parsing_lib/inc/convert.h>
#include <parsing_lib/inc/data_types.h>

#include <elf/reader32.hpp>

#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Example usage is ./act-kernels-elf-poc <path-to-elf>" << '\n';
        return 1;
    }
    
    std::ifstream stream(argv[1], std::ios::binary);
    std::vector<char> elfBlob((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    stream.close();

    vpux::VPUIP::ELFBlobSerializer blobSerializer;

    blobSerializer.initActKernel(elfBlob, "hswish");

    blobSerializer.addActKernel();

    // auto in_out_size = 256 * 256 * 1; // iw * ih * ic

    // float input[in_out_size];
    // float output[in_out_size];

    // for (int i = 0; i < 256*256*1; i++){
    //     input[i] = 0;
    //     output[i] = 0;
    // }

    // uint32_t tensor_size = 256 * 56;


    blobSerializer.addActInvocation();
    // blobSerializer.addActInvocation();

    blobSerializer.finalizeActKernelWrappers();

    const auto final_elf = blobSerializer.getBlob();

    std::ofstream out_stream("actKernel_poc.elf", std::ios::out | std::ios::binary);
    out_stream.write(final_elf.data(), final_elf.size());

    return 0;
}

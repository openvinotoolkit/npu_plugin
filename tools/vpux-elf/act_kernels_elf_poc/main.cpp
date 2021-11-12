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

    blobSerializer.addActInvocation();
    blobSerializer.addActInvocation();

    blobSerializer.finalizeActKernelWrappers();


    DmaWrapper dma{};

    parsing_lib::DMATask t;
    t.src.data_dtype = parsing_lib::DType::U8;
    t.dst.data_dtype = parsing_lib::DType::U8;
    t.src.dimensions = std::vector<uint32_t>({1, 320, 7, 7});
    t.dst.dimensions = std::vector<uint32_t>({1, 320, 7, 7});
    t.src.strides = std::vector<float>({1.f, 15680.f, 1.f, 2240.f, 320.f});
    t.dst.strides = std::vector<float>({1.f, 15680.f, 1.f, 2240.f, 320.f});
    t.src.order = 0x1342;
    t.dst.order = 0x1342;
    t.compression = false;

    parsing_lib::convertDmaTask(t, dma.transaction);

    vpux::VPUIP::DMATaskExtension dmaTaskExtension{};
    dmaTaskExtension.input.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
    dmaTaskExtension.input.location.locationIndex = 0;
    dmaTaskExtension.input.offset = 0;

    dmaTaskExtension.output.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableOutput;
    dmaTaskExtension.output.location.locationIndex = 0;
    dmaTaskExtension.output.offset = 0;

    ResourceRequirements resourceRequirements{};
    resourceRequirements.barrier_count = 0;
    resourceRequirements.slice_count = 2;

    mlir::MLIRContext context;

    const mlir::SmallVector<mlir::MemRefType> inputs{mlir::MemRefType::get(
            {1, 320, 7, 7}, mlir::IntegerType::get(&context, 8, mlir::IntegerType::SignednessSemantics::Unsigned))};

    blobSerializer.setNetworkInputs(inputs);
    blobSerializer.setNetworkOutputs(inputs);

    blobSerializer.setDDRScratch(0);
    blobSerializer.setResourceRequirements(resourceRequirements);
    blobSerializer.setDMATasks0({{dma, dmaTaskExtension}});

    blobSerializer.write("act_kernel_blob.elf");

    return 0;
}

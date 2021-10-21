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

int main() {
    vpux::VPUIP::ELFBlobSerializer blobSerializer;
    vpux::VPUIP::DmaTask dmaTask;

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

    parsing_lib::convertDmaTask(t, dmaTask.dmaDescriptor.transaction);

    ResourceRequirements resourceRequirements{};
    resourceRequirements.barrier_count = 1;
    resourceRequirements.slice_count = 2;

    mlir::MLIRContext context;
    const mlir::SmallVector<mlir::MemRefType> inputs{mlir::MemRefType::get(
            {1, 320, 7, 7}, mlir::IntegerType::get(&context, 8, mlir::IntegerType::SignednessSemantics::Unsigned))};

    blobSerializer.setNetworkInputs(inputs);
    blobSerializer.setNetworkOutputs(inputs);

    blobSerializer.setDDRScratch(15680);
    blobSerializer.setResourceRequirements(resourceRequirements);
    blobSerializer.setLeadingDMACount({2});

    auto dmaTask2 = dmaTask;
    dmaTask.dmaDescriptor.transaction.barriers.prod_mask = 1;
    dmaTask2.dmaDescriptor.transaction.barriers.cons_mask = 1;

    dmaTask.input.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
    dmaTask.input.location.locationIndex = 0;
    dmaTask.input.offset = 0;

    dmaTask.output.location.memLocation = vpux::VPUIP::MemoryLocation::VPU_DDR_BSS;
    dmaTask.output.location.locationIndex = 0;
    dmaTask.output.offset = 0;

    dmaTask2.input.location.memLocation = vpux::VPUIP::MemoryLocation::VPU_DDR_BSS;
    dmaTask2.input.location.locationIndex = 0;
    dmaTask2.input.offset = 0;

    dmaTask2.output.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableOutput;
    dmaTask2.output.location.locationIndex = 0;
    dmaTask2.output.offset = 0;

    mlir::SmallVector<BarrierWrapper> barrierConfigs{BarrierWrapper{-1, 1, 1, 0}};

    blobSerializer.setDMATasks({dmaTask, dmaTask2});
    blobSerializer.setBarrierConfigs(barrierConfigs);

    blobSerializer.write("nn_blob.elf");

    return 0;
}

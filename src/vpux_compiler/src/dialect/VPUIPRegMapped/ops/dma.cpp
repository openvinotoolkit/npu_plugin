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

//#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include "llvm/Support/Debug.h"  // Alex

using namespace vpux;

//
// UPADMAOp
//

/*
// Alex
void vpux::VPUIPRegMapped::UPADMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value src,
                                  mlir::Value dst) {
    build(builder, state, src, dst, mlir::ValueRange{}, mlir::ValueRange{});
}
*/

// VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UPADMAOp::serialize(VPUIP::BlobWriter& writer) {
void vpux::VPUIPRegMapped::UPADMAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto inputOff = writer.getTensor(input());
    const auto outputOff = writer.getTensor(output());

    MVCNN::UPADMATaskBuilder builder(writer);
    builder.add_src(inputOff);
    builder.add_dst(outputOff);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPADMATask};
    */

    (void)buffer;
}

//
// NNDMAOp
//

/*
// Alex
void vpux::VPUIPRegMapped::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value src,
                                 mlir::Value dst) {
    build(builder, state, src, dst, mlir::ValueRange{}, mlir::ValueRange{}, false);
}
*/

/*
// Bit field for fine-grained configuration of CMXDMA transaction
typedef struct __attribute__ ((packed)) {
    uint32_t type              : 2;   // Transaction type(1D/2D)
    uint32_t burst_length      : 8;   // Burst length
    uint32_t critical          : 1;   // Critical task
    uint32_t interrupt_en      : 1;   // Interrupt enable
    uint32_t interrupt_trigger : 7;   // Interrupt status id when task is executed
    uint32_t skip_nr           : 7;   // Skip descriptor
    uint32_t order_forced      : 1;   // Force ordering. Dispatch the current
                                      //   task only after the previous task
                                      //   has completed
    uint32_t watermark_en      : 1;   // Transaction watermark enable
    uint32_t huf_en            : 1;   // Huffman decompressor enable
    uint32_t barrier_en        : 1;   // Barrier use enable
    uint64_t reserved          : 34;  // Reserved
} HglCmxDmaConfigBits;

    IMPORTANT NOTE:
            type 3D transaction
            burst_length UI8Attr – este 8
            critical UI1Attr – este Nu
            interrupt_en UI1Attr - Yes
            //interrupt_trigger UI7Attr -
            skip_nr UI7Attr – nu avem,
            order_forced UI1Attr – Yes,
            watermark_en UI1Attr - Yes
            huf_en UI1Attr – No
>;

// From sipp/arch/3600/src/leon/sippCmxDmaIf.h:
typedef struct {
    uint32_t src_width;     // Bytes of data required from one line of source
    int32_t src_stride;     // Length in bytes from start of one line of data,
                            // to start of next line of data
    uint32_t dst_width;     // Bytes of data required from one line of destination
    int32_t dst_stride;     // Length in bytes from start of one line of data,
                            // to start of next line of data
} HglCmxDma2DAttributes;


struct BarrierUserConfig {
    PhysicalBarrierMask wait_mask_;
    PhysicalBarrierMask post_mask_;
    unsigned short start_after_;
    unsigned short clean_after_;
    unsigned int virtual_dep_;
};


// From HglCmxDmaBarrierCfg
typedef struct {
    uint64_t prod_mask; // 64-bit mask depicting which barriers are affected by
                        // task completion
    uint64_t cons_mask; // 64-bit mask depicting which barriers are gating the current
                        // Link Agent
} HglCmxDmaBarrierCfg;


// From sipp/arch/3600/src/leon/sippCmxDmaIf.h
// This struct corresponds to "Table 984: 2D Striding Transaction Command with
//   Planes" (and a bit Table 986) from "Section 8.5.5.1 Descriptor for 2D Block
//   Transfers" from the main Databook document of KeemBay - you read top-down,
//   right to left.
//
// Generic transaction type
typedef struct __attribute__((packed)) {
    uint64_t link_address : 40;  // pointer to the next element in linked list
    uint32_t reserved     : 23;
    uint32_t watermark    :  1;  // watermark to indicate that the transaction has completed
    union {
        HglCmxDmaConfigBits cfg_bits;
        uint64_t full_cfg_register;
    } cfg_link;
    uint64_t src;               // Address of the data transfer source
    uint64_t dst;               // Address of the data transfer destination
    uint32_t length;            // Transaction length
    uint32_t num_planes   :  8; // Number of planes
    uint32_t task_id      : 24; // Task id for the current transaction
    int32_t src_plane_stride;   // Source plane stride
    int32_t dst_plane_stride;   // Destination plane stride
    union {
        HglCmxDma2DAttributes attr2d;   // Attributes that apply for 2D transactions (i.e. striding)
        HglCmxDmaBarrierCfg barriers1d; // Barrier mask configurations for 1D transactions
    };
    HglCmxDmaBarrierCfg barriers; // Barrier mask configurations for 2D transactions
    // The descriptor must be aligned to 64 byte boundary
    // This is needed for L2 cache line alignment
} HglCmxDmaTransaction __attribute__((aligned(HGL_CPU_L2CACHE_ALIGNMENT)));


// TODO: Figure out why do we have 38 bits for LINK ADDRESS as seen at
https://docs.intel.com/documents/Movidiusinternal/vpu27/Common/HW/VPU_HAS.html#descriptor-for-2d-block-transfers
*/
// VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NNDMAOp::serialize(VPUIP::BlobWriter& writer) {
// VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NNDMAOp::serialize() {
/// void vpux::VPUIPRegMapped::NNDMAOp::serialize() {
/// char* vpux::VPUIPRegMapped::NNDMAOp::serialize(int& resBufferSize) {
void vpux::VPUIPRegMapped::NNDMAOp::serialize(std::vector<char>& buffer) {
    /*
    const auto srcOff = writer.getTensor(input());
    const auto dstOff = writer.getTensor(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_compression(compression());
    builder.add_port(checked_cast<uint8_t>(port()));
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
    */

    /*
    Write the hardware register mapped structure HglCmxDmaTransaction
    For this we need to write the following fields from HglCmxDmaTransaction:
        - uint64_t link_address : 40;  // pointer to the next element in linked list
        - uint32_t reserved     : 23;
        - uint32_t watermark    :  1;  // watermark to indicate that the transaction has completed

        // HglCmxDmaConfigBits
        type - UI2Attr - 3D transaction
        burst_length UI8Attr - este 8
        critical UI1Attr - este Nu
        interrupt_en UI1Attr - Yes
        interrupt_trigger UI7Attr - all set to 0 // exec ar trebui sa aiba intrerupere
        skip_nr UI7Attr - nu avem,
        order_forced UI1Attr - Yes,
        watermark_en UI1Attr - Yes
        huf_en UI1Attr - No
        uint32_t barrier_en        : 1;   // Barrier use enable - ...
        uint64_t reserved          : 34;  // Reserved - ...
        // The number of bits of struct HglCmxDmaConfigBits: 2 + 8 + 1 + 1 + 7 + 7 + 1 + 1 + 1 + 1 + 34 = 64 bits

        uint64_t src;               // Address of the data transfer source
        uint64_t dst;               // Address of the data transfer destination
        uint32_t length;            // Transaction length
        uint32_t num_planes   :  8; // Number of planes
        uint32_t task_id      : 24; // Task id for the current transaction
        int32_t src_plane_stride;   // Source plane stride
        int32_t dst_plane_stride;   // Destination plane stride

        // HglCmxDmaBarrierCfg barriers1d;
        uint64_t prod_mask; // 64-bit mask depicting which barriers are affected by
                            // task completion
        uint64_t cons_mask; // 64-bit mask depicting which barriers are gating the current
                            // Link Agent

        // HglCmxDmaBarrierCfg barriers;
        uint64_t prod_mask; // 64-bit mask depicting which barriers are affected by
                            // task completion
        uint64_t cons_mask; // 64-bit mask depicting which barriers are gating the current
                            // Link Agent

    The size of the struct is:
      64 + 64 + 64 + 64 + 32 + 8+24 + 32 + 32 + 64 + 64 + 64 + 64 = 640 bits
    */

    llvm::dbgs() << "Alex: Entered void vpux::VPUIPRegMapped::NNDMAOp::serialize()\n";

    llvm::dbgs() << "  input = " << input() << "\n";
    llvm::dbgs() << "  output_buff = " << output_buff() << "\n";

    llvm::dbgs() << "  start_after = " << start_after() << "\n";

    llvm::dbgs() << "Alex: Exiting void vpux::VPUIPRegMapped::NNDMAOp::serialize()\n";

    //(void)buffer;
    /*
    for (int i = 0; i < 640 / 8; i++) {
        buffer.push_back(i);
    }
    */

    /*
    // uint64_t link_address : 40;
    uint64_t link_address = 0 & 0xFFFFFFFFFF;
    buffer.push_back(link_address & 0xFF);
    buffer.push_back((link_address >> 8) & 0xFF);
    buffer.push_back((link_address >> 16) & 0xFF);
    buffer.push_back((link_address >> 24) & 0xFF);
    buffer.push_back((link_address >> 32) & 0xFF);

    // uint32_t reserved     : 23;
    // uint32_t watermark    :  1;
    uint32_t reserved = 0 & 0x7FFFFF;
    uint32_t watermark = 0 & 0x1;
    int tmp = (reserved << 1) | watermark;
    buffer.push_back(link_address & 0xFF);
    buffer.push_back((link_address >> 8) & 0xFF);
    buffer.push_back((link_address >> 16) & 0xFF);
    */

    // Bit field for fine-grained configuration of CMXDMA transaction
    typedef struct __attribute__((packed)) {
        uint32_t type : 2;               // Transaction type(1D/2D)
        uint32_t burst_length : 8;       // Burst length
        uint32_t critical : 1;           // Critical task
        uint32_t interrupt_en : 1;       // Interrupt enable
        uint32_t interrupt_trigger : 7;  // Interrupt status id when task is executed
        uint32_t skip_nr : 7;            // Skip descriptor
        uint32_t order_forced : 1;       // Force ordering. Dispatch the current
                                         //   task only after the previous task
                                         //   has completed
        uint32_t watermark_en : 1;       // Transaction watermark enable
        uint32_t huf_en : 1;             // Huffman decompressor enable
        uint32_t barrier_en : 1;         // Barrier use enable
        uint64_t reserved : 34;          // Reserved
    } HglCmxDmaConfigBits;

    typedef struct {
        uint64_t prod_mask;  // 64-bit mask depicting which barriers are affected by
                             // task completion
        uint64_t cons_mask;  // 64-bit mask depicting which barriers are gating the current
                             // Link Agent
    } HglCmxDmaBarrierCfg;

    // From sipp/arch/3600/src/leon/sippCmxDmaIf.h
    // This struct corresponds to "Table 984: 2D Striding Transaction Command with
    //   Planes" (and a bit Table 986) from "Section 8.5.5.1 Descriptor for 2D Block
    //   Transfers" from the main Databook document of KeemBay - you read top-down,
    //   right to left.
    //
    // Generic transaction type
    typedef struct __attribute__((packed)) {
        uint64_t link_address : 40;  // pointer to the next element in linked list
        uint32_t reserved : 23;
        uint32_t watermark : 1;  // watermark to indicate that the transaction has completed
        union {
            HglCmxDmaConfigBits cfg_bits;
            uint64_t full_cfg_register;
        } cfg_link;
        uint64_t src;              // Address of the data transfer source
        uint64_t dst;              // Address of the data transfer destination
        uint32_t length;           // Transaction length
        uint32_t num_planes : 8;   // Number of planes
        uint32_t task_id : 24;     // Task id for the current transaction
        int32_t src_plane_stride;  // Source plane stride
        int32_t dst_plane_stride;  // Destination plane stride

        // Alex: union {
        // Alex:   HglCmxDma2DAttributes attr2d;   // Attributes that apply for 2D transactions (i.e. striding)
        HglCmxDmaBarrierCfg barriers1d;  // Barrier mask configurations for 1D transactions
        //};

        HglCmxDmaBarrierCfg barriers;  // Barrier mask configurations for 2D transactions
        // The descriptor must be aligned to 64 byte boundary
        // This is needed for L2 cache line alignment
    } HglCmxDmaTransaction;  // __attribute__((aligned(HGL_CPU_L2CACHE_ALIGNMENT)));

    HglCmxDmaTransaction tmp;

    /*
    // IMPORTANT NOTE: input() returns mlir::Value, which is NOT constant.
    tmp.src = input();

    tmp.dst = output_buff();
    */

    // tmp.link_address = 0 & 0xFFFFFFFFFF; // TODO
    // uint32_t reserved = 0 & 0x7FFFFF; // TODO
    // uint32_t watermark = 0 & 0x1; // TODO

    tmp.cfg_link.cfg_bits.burst_length = 8 & 0xFF;
    tmp.cfg_link.cfg_bits.critical = 1 & 0x1;
    tmp.cfg_link.cfg_bits.interrupt_en = 1 & 0x1;
    tmp.cfg_link.cfg_bits.interrupt_trigger = 1 & 0x7F;  // TODO: put correct value
    tmp.cfg_link.cfg_bits.skip_nr = 1 & 0x7F;            // TODO put correct value
    tmp.cfg_link.cfg_bits.order_forced = 1 & 0x1;
    tmp.cfg_link.cfg_bits.watermark_en = 1 & 0x1;
    tmp.cfg_link.cfg_bits.huf_en = 0 & 0x1;
    tmp.cfg_link.cfg_bits.barrier_en = 0 & 0x1;  // TODO put correct value
    tmp.cfg_link.cfg_bits.reserved = 0 & 0x1;    // TODO put correct value

    // TODO: more processing to be done

    char* ptrCharTmp = (char*)(&tmp);
    for (long unsigned i = 0; i < sizeof(HglCmxDmaTransaction); i++) {
        buffer.push_back(*(ptrCharTmp + i));
    }
}

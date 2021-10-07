//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

namespace vpux {
namespace VPUIP {

unsigned int swizzle_addr(unsigned int addr, unsigned char key){
    // Some constant get automatically optimized by the compiler
    const unsigned int LOG2_RAM_CUT_BYTES = 9; // address size of 32 KB RAM cut with 128 bits words
    const unsigned int CUT_ADDRESS_MASK_10b = (1 << 11) - 1; // RAM cut address mask
    const unsigned int MAX_SWIZZLE_KEY = 5;
    const unsigned int RAM_CUT_ADDRESS_MASK = (1 << LOG2_RAM_CUT_BYTES) - 1;

    unsigned int stagger_address_mask = (1 << key) - 1;
    int shift = LOG2_RAM_CUT_BYTES - key;
    unsigned int addr_stagger, phy_addr;

    addr_stagger = (addr >> 4) & CUT_ADDRESS_MASK_10b; // get the address in the ramcut
    addr_stagger = addr_stagger >> MAX_SWIZZLE_KEY; // right shift 5 bits
    addr_stagger = addr_stagger & stagger_address_mask; // get only the relevant bits of the address
    addr_stagger = addr_stagger << shift; // Shift them back the their original locaiton

    phy_addr = addr + addr_stagger;
    phy_addr = phy_addr & RAM_CUT_ADDRESS_MASK;
    phy_addr = phy_addr + (addr & (~RAM_CUT_ADDRESS_MASK));

    return phy_addr;
}

template <class T>
void swizzled_img(T* input_array, T* output_array, const unsigned char key, const unsigned int size){
    for (unsigned int addr = 0; addr < size; addr++ ){
        output_array[swizzle_addr(addr, key)] = input_array[addr];
    }
}

template <class T>
int swizzled_size(T*, const unsigned char key, const unsigned int size){
    unsigned int max_size = 0;
    for (unsigned int addr = 0; addr < size; addr++ ){
        max_size = std::max(max_size, swizzle_addr(addr, key));
    }
    // Align to 64 bytes
    return (((max_size + 63) / 64) * 64);
}

}  // namespace VPUIP
}  // namespace vpux

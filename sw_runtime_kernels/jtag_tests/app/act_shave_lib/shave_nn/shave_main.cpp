///
/// @file
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Simple shave function which reads 2 numbers from NNCMX, and writes sum to DRAM.
///            Test memory accesses.
///

#include <stdlib.h>
#include <mv_types.h>
#include <registersVpu3600.h>
#include "descriptor.h"
#include "act_shave_mgr.h"
#include <sw_shave_lib_common.h>
#include <stdio.h>

//unsigned char __attribute__((section(".data"), aligned(64))) sParam[SHAVE_LIB_PARAM_SIZE];
//unsigned char __attribute__((section(".data"), aligned(64))) sData[SHAVE_LIB_DATA_SIZE];
extern unsigned char actShaveParam[];

#include "sw_shave_res_manager.h"
#include "sw_layer.h"
// #include "BMsvuCommonShave.h"
//#include "../runtime/common_functions.h"
//#include "../runtime/barrier.h"

//__attribute__((noreturn)) extern void _exit(int);

/// @brief Shave halt instruction
#define SHAVE_HALT __asm volatile ( \
    "NOP"              "\n\t" \
    "BRU.swih 0x001F"  "\n\t" \
    "NOP 6"            "\n\t" \
    ::: "memory");


uint32_t read(uint32_t adr) {
    return *(uint32_t *)(adr);
}

void write(uint32_t adr, uint32_t val) {
    *(uint32_t *)(adr) = val;
    return;
}

uint32_t get_window_address(uint32_t window_number) {
    uint32_t win_addr = 0;
    switch (window_number) {
        case 0:
            asm volatile("lsu0.lda.32 %[addr], SHAVE_LOCAL, 0x10" : [addr] "=r"(win_addr));
            break;
        case 1:
            asm volatile("lsu0.lda.32 %[addr], SHAVE_LOCAL, 0x14" : [addr] "=r"(win_addr));
            break;
        case 2:
            asm volatile("lsu0.lda.32 %[addr], SHAVE_LOCAL, 0x18" : [addr] "=r"(win_addr));
            break;
        case 3:
            asm volatile("lsu0.lda.32 %[addr], SHAVE_LOCAL, 0x1c" : [addr] "=r"(win_addr));
            break;
    }
    return win_addr;
}

void set_window_address(uint32_t window_number, uint32_t win_addr) {
    switch (window_number) {
        case 0:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x10" ::[addr] "r"(win_addr));
            asm volatile("nop 5");
            break;
        case 1:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x14" ::[addr] "r"(win_addr));
            asm volatile("nop 5");
            break;
        case 2:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x18" ::[addr] "r"(win_addr));
            asm volatile("nop 5");
            break;
        case 3:
            asm volatile("lsu0.sta.32 %[addr], SHAVE_LOCAL, 0x1c" ::[addr] "r"(win_addr));
            asm volatile("nop 5");
            break;
    }
}

void fifoWaitGpio() {
    // SNN    fifo0 bit24 (shave nn monitor only)
    // SNN    fifo1 bit25 (shave nn monitor only)
    // AS     fifo3 bit24 (act-shave monitor only)
    // SNN+AS fifo2 bit0
    // const unsigned int fifoEmptyBit = fifo != 2 ? (1u << (24 + (fifo == 1))) : 1u;
    const unsigned int fifoEmptyBit = 1U << 24;

    asm volatile("CMU.CMTI.BITP %[fifo_empty_bit], P_GPI"
                 "\n\t"
                 "PEU.PCXC NEQ 0 || BRU.RPIM 0 || CMU.CMTI.BITP %[fifo_empty_bit], P_GPI"
                 "\n\t"
    :
    : [fifo_empty_bit] "r"(fifoEmptyBit)
    : "memory");
}

extern "C"
void act_shave_runtime_shaveMain() {
//    uint32_t win_c_address = get_window_address(0);
    uint32_t win_d_address = get_window_address(1);
//    uint32_t win_e_address = get_window_address(2);
//    uint32_t win_f_address = get_window_address(3);

    //set_window_address(0, win_a_address);
    //printf("Hello from act-shave\n");

//    const uint32_t kernel_entry_addr = 0x1E000000;

    win_d_address = 0x2E000000;
    shv_job_header* job_ptr = 0;
//    set_window_address(1, win_d_address);

    while(1) {
        fifoWaitGpio();
        uint32_t fifo_val = read(NNCMX_NCE_FIFO_3_0_ADR);
        write(0x2e1100A0, fifo_val);

        if(!fifo_val) {
            break;
        }
        if (0xFFFFFFFF == fifo_val)
            continue;


        job_ptr = (shv_job_header*)fifo_val;

        //      DEBUG pointers
        write(0x2e1100A4, job_ptr->shv_kernel_address);
        write(0x2e1100A8, job_ptr->shv_data_address);
        write(0x2e1100AC, job_ptr->shv_pre_address);
        write(0x2e1100B0, job_ptr->kernel_arg_address);
        write(0x2e1100B4, (uint32_t)((nn::shave_lib::SoftParams *)job_ptr->kernel_arg_address)->layerParams);
      //  write(0x2e1100B8, job_ptr->aba_pointer);

//        write(0x2e1100BC, win_f_address);


        // L2 cache prefetch
  //      set_window_address(2, win_e_address);
  //      set_window_address(3, win_f_address);

//        wait_for_barrier(job_ptr->wait_barrier_mask);

        // parameters conversion from softparams to kernel specific - using preambula
        auto pre_func = reinterpret_cast<void (*)(const nn::shave_lib::LayerParams *, nn::shave_lib::ShaveResourceManager *)>(job_ptr->shv_pre_address);
        if (pre_func) {
            auto& shaveResMgr = ACTShaveManager::instance();
            auto absolutePointers = *reinterpret_cast<nn::shave_lib::AbsoluteAddresses*>(job_ptr->aba_pointer);
            shaveResMgr.setAbsolutePointers(absolutePointers);
            (*pre_func)((nn::shave_lib::LayerParams*)job_ptr->kernel_arg_address, &shaveResMgr);
        }

            // execute kernel
        auto kernel_func = reinterpret_cast<void (*)(void *)>(job_ptr->shv_kernel_address);
        if (kernel_func) {
            set_window_address(1, job_ptr->shv_kernel_address);
            (*kernel_func)(reinterpret_cast<void *>(actShaveParam));
        }

            // decrement_wait_barriers(job_ptr->wait_barrier_mask);
            // decrement_update_barriers(job_ptr->update_barrier_mask);

//        }
        if(job_ptr->job_completed_pointer){
            *reinterpret_cast<uint32_t*>(job_ptr->job_completed_pointer) = 1;
        }
//        continue;
    }

//    write(0x2e1100B0, win_c_address);
//    write(0x2e1100B4, win_d_address);
//    write(0x2e1100B8, win_e_address);
//    write(0x2e1100BC, win_f_address);

    SHAVE_HALT;
}

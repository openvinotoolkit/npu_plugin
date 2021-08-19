#include "common_functions.h"

#ifndef ADDRESS_H
#define ADDRESS_H
#define DRAM_BASE_ADDR 0x80000000
#define CMX0_BASE_ADDR 0x2E000000
#define CMX1_BASE_ADDR 0x2E200000
#endif

#ifdef _WIN32
extern uint8_t DRAM_MEMORY_BUFFER[];
extern uint8_t CMX_MEMORY_BUFFER[];

void retarget_address(uint64_t *addr) {
  if (*addr >= DRAM_BASE_ADDR) {
    *addr -= DRAM_BASE_ADDR;
    *addr += (uint64_t)DRAM_MEMORY_BUFFER;
  }
  else if (*addr >= CMX1_BASE_ADDR) {
    *addr -= CMX1_BASE_ADDR;
    *addr += (uint64_t)CMX_MEMORY_BUFFER;
  }
  else if (*addr >= CMX0_BASE_ADDR) {
    *addr -= CMX0_BASE_ADDR;
    *addr += (uint64_t)CMX_MEMORY_BUFFER;
  }
}
#else
void retarget_address(uint64_t */*addr*/) {
}
#endif

void mem_write_u64(uint64_t base_addr, uint64_t offset, uint64_t data){
#ifdef _WIN32
    return;
#endif
    volatile uint64_t* p_addr = reinterpret_cast<volatile uint64_t*>(base_addr + offset);
    *p_addr = data;
}

void mem_read_u64(uint64_t base_addr, uint64_t offset, uint64_t *p_data){
#ifdef _WIN32
    return;
#endif
    volatile uint64_t* p_addr = reinterpret_cast<volatile uint64_t*>(base_addr + offset);
    *p_data = *p_addr;
}

void mem_write_u32(uint64_t base_addr, uint64_t offset, uint32_t data) {
#ifdef _WIN32
    return;
#endif
    volatile uint32_t* p_addr = reinterpret_cast<volatile uint32_t*>(base_addr + offset);
    *p_addr = data;
}

void mem_read_u32(uint64_t base_addr, uint64_t offset, uint32_t *p_data) {
#ifdef _WIN32
    return;
#endif
    volatile uint32_t* p_addr = reinterpret_cast<volatile uint32_t*>(base_addr + offset);
    *p_data = *p_addr;
}

void wait_n_cycles(uint32_t n){
    for(uint32_t cycle = 0; cycle < n; cycle++)
        ;
}

void cpu_memcpy(uint8_t* src_ptr, uint8_t* dst_ptr, uint32_t len){
    for(uint32_t i = 0; i < len; ++i)
    {
        *dst_ptr++ = *src_ptr++;
    }
}

#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H
#include <stdint.h>

void mem_write_u64(uint64_t base_addr, uint64_t offset, uint64_t data);
void mem_read_u64(uint64_t base_addr, uint64_t offset, uint64_t *p_data);
void mem_write_u32(uint64_t base_addr, uint64_t offset, uint32_t data);
void mem_read_u32(uint64_t base_addr, uint64_t offset, uint32_t *p_data);
void wait_n_cycles(uint32_t n);
void cpu_memcpy(uint8_t* src_ptr, uint8_t* dst_ptr, uint32_t len);
void retarget_address(uint64_t *addr);
#endif
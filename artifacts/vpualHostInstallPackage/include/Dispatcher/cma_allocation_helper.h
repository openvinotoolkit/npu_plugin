// {% copyright %}
///
/// @file      cma_allocation_helper.h
///

#ifndef __CMA_ALLOCATION_HELPER_H__
#define __CMA_ALLOCATION_HELPER_H__

#include <string>

/** CMA allocation helper class. */
class CmaData {
  public:
    int            fd;          					/*< File descriptor. */
    bool		   b_use_vpusmm;
    unsigned char* buf;         					/*< Buffer for use in virtual address space. */
    unsigned long  phys_addr;   					/*< Physical address of allocation. */
    unsigned int   size;        					/*< Size of allocation. */
    uint32_t       device_id;

    int Create(std::string);    					/*< Open and map the allocation files. */
    int Create(size_t size);
    CmaData(uint32_t device_id = 0)
        : fd { -1 }
        , b_use_vpusmm { false }
        , buf { nullptr }
        , phys_addr { 0 }
        , size { 0 }
        , device_id(device_id)
        {};
    ~CmaData();                 					/*< Close the file if opened. */

    // Delete copy constructor and assignment operator.
    CmaData(const CmaData&) = delete;
    CmaData& operator=(const CmaData&) = delete;

};

#endif // __CMA_ALLOCATION_HELPER_H__

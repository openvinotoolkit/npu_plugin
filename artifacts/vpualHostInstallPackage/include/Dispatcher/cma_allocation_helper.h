///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
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
    unsigned char* buf;         					/*< Buffer for use in virtual address space. */
    unsigned long  phys_addr;   					/*< Physical address of allocation. */
    unsigned int   size;        					/*< Size of allocation. */
  	uint32_t       device_id;


    int Create(std::string);    					/*< Open and map the allocation files. */
    int Create(size_t size);
    CmaData(uint32_t device_id)
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
private:
	bool           b_use_vpusmm;

};

#endif // __CMA_ALLOCATION_HELPER_H__

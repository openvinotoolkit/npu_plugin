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
/// @file      VpuData.h
/// 

#ifndef __VPU_DATA_H__
#define __VPU_DATA_H__

#include <stdint.h>
#include <stddef.h>

// TODO ideally we would have this defined somewhere else? not sure where is best though.. perhaps XLinkUAPI?
#ifdef __REMOTE_HOST__
typedef uint8_t* DevicePtr;
#else
typedef uint32_t DevicePtr;
#endif

/**
 * Data handle for data to be shared with VPU.
 *
 * Abstraction should allow common api between local and remote host
 * applications.
 */
class VpuData {
  private:

#ifdef __REMOTE_HOST__
    /* Remote host solely uses virtual memory. */
#else
    /* Local host must manage CMA memory. */
    unsigned long  phys_addr_;  /*< Physical address of allocation. */
    int            fd_;         /*< File descriptor for vpusmm driver. */
#endif // Host type

    size_t         size_;       /*< Size of allocation. */
    unsigned char* buf_;        /*< Buffer for use in virtual address space. */
  	uint32_t       device_id_;

  public:

    // Create a VPU data object of the given size.
    VpuData(size_t size, uint32_t device_id = 0);
    // TODO - Maybe we also need this sort of stuff? Can add later.
    // VpuData(size_t size, alignement = 64U, zero_initialise=false);

    ~VpuData();

    // Implicit type conversions
#ifdef __REMOTE_HOST__
    operator DevicePtr() const { return buf_; }
#else
    operator DevicePtr() const { return phys_addr_; }
    operator unsigned char*() const { return buf_; }
#endif
    operator void*() const { return buf_; }

    // Delete copy constructor and assignment operator.
    VpuData(const VpuData&) = delete;
    VpuData& operator=(const VpuData&) = delete;

    // Size getter.
    size_t size() const { return size_; }
};

#endif // __VPU_DATA_H__

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
/// @file      MemAllocator.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Pool allocator over VPUAL.
///

#ifndef __MEM_ALLOCATOR_H__
#define __MEM_ALLOCATOR_H__

#include <stdint.h>

#include "Flic.h"
#include "VpualDispatcher.h"

//##############################################################

class IAllocator : public VpualStub {
  public:
    /** Constructor just invokes the parent constructor. */
    IAllocator(std::string type) : VpualStub(type){};
};

//##############################################################

class RgnAllocator: public IAllocator {
  public:
  //some of the VideoEncode buffs require 32Byte alignment
   RgnAllocator() : IAllocator("RgnAllocator") {};
   // Note the Create function takes a physical base address (uint32_t) rather than (void *)
   void  Create(uint32_t pBaseAddr, uint32_t sz, uint32_t alignment = 64);
   void  Delete();

   // The following functions won't be exposed on the Host.
   //    void *Alloc(size_t size);
   //    void  Free (void *ptr);
};

//##############################################################

class HeapAllocator : public IAllocator {
 public:
    // Note we have explicitly listed the datatype as uint32_t, rather than size_t
    HeapAllocator(uint32_t alignment = 64);

    // The following functions won't be exposed on the Host.
    //    void *Alloc(size_t size);
    //    void  Free (void *ptr);
};

//##############################################################

//Stock objects
// extern uint8_t       RgnBuff[];
// extern uint8_t       RgnBuffCMX[];

// TODO Default pool size:
#define DEF_POOL_SZ (8*800*480*3/2)

#endif // __MEM_ALLOCATOR_H__

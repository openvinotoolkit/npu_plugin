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
    IAllocator(std::string type, uint32_t device_id) : VpualStub(type, device_id){};
};

//##############################################################

class RgnAllocator: public IAllocator {
  public:
  //some of the VideoEncode buffs require 32Byte alignment
   RgnAllocator(uint32_t device_id = 0) : IAllocator("RgnAllocator", device_id) {};
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
    HeapAllocator(uint32_t alignment = 64, uint32_t device_id = 0);

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

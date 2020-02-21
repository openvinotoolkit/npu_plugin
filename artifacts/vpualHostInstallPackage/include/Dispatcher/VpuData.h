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

  public:

    // Create a VPU data object of the given size.
    VpuData(size_t size);
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

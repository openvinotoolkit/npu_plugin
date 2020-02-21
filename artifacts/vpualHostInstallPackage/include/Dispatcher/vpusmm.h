/*
Copyright (C) 2019 Intel Corporation

SPDX-License-Identifier: MIT
*/

#ifndef _VPUSMM_H_
#define _VPUSMM_H_
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Vpu-shared memory type.*/
enum VPUSMMType {
VPUSMMTYPE_COHERENT = 0x0000,
/**< allocate coherent memory, it's the default
     - cache coherent
     - using non-cachable(writecombine) mapping on CPU side
     - no explicit cache flush/invalidate is required
     - low performance for CPU side access
*/
VPUSMMTYPE_NON_COHERENT = 0x0100, /**< allocate non-coherent memory, non-default
 	- cache non-coherent
 	- using cached mapping on CPU side
 	- explicit sync operation is required
 	- high performace for CPU side access
	*/
};

enum VPUAccessType {
VPU_DEFAULT = 0x00,	// this has the same meaning as VPU_RW
VPU_READ  = 0x01,
VPU_WRITE = 0x02,
VPU_RW    = 0x03,	//VPU_READ | VPU_WRITE
VPU_WR    = 0x03,	//VPU_READ | VPU_WRITE
};

/** allocate a VPU-shared buffer object
 * @param size [IN]: size in bytes, must be multiple of page size, otherwise mmap syscall may fail
 * @param type [IN]: vpu-shared memory type
 * @return: the file descriptor of the underlying vpu-shared buffer object
 *          or <0 if failed
 *
 * user is supposed to call standard UNIX API close() on the fd
 * when not using it anymore, failing doing so will cause leakage
 * during runtime but will be cleaned when application process
 * ceased execution.
 *
 */
int vpusmm_alloc_dmabuf(unsigned long size, enum VPUSMMType type);


/**  Import an external DMABuf object into VPU device for accessing on VPU side
* @param dmabuf_fd [IN] : the DMABuf fd exported by other driver(codec/camera/...) or our driver
* @param vpu_access [IN] : how VPU is going to access the buffer (readonly or writeonly or readwrite)
*                          application should provide this param as accurate as possible, use VPU_DEFAULT
*                          when not sure, this mainly affect cache coherency
* @return a non-zero VPU-side address corresponds to the DMABuf which suitable for VPU to access.
*         zero if the DMABuf is not suitable for VPU to access (for example, due to not physical contiguous)
*         in this case, user should allocate another DMABuf with VPUSMM and do copy by CPU
*
* import operation is vital for VPU to access the buffer because:
*   1. necessary mapping resource is setup and maintained during import for VPU to access the buffer
*   2. the physical pages is pinned during import so no page fault will happen
*   3. kernel rb-tree is setup and maintained so virt-addr can be translated into device address
*/
unsigned long vpusmm_import_dmabuf(int dmabuf_fd, enum VPUAccessType vpu_access);


/**  Unimport an imported external DMABuf when VPU side no longer accesses it
* @param dmabuf_fd [IN] : the DMABuf fd to be unimported
*
* after this function call, the reference to the DMABuf will be freed and the VPU address should not
* be considered to be valid any more, so call this function only after VPU-side access has done.
*/
void vpusmm_unimport_dmabuf(int dmabuf_fd);

/**  get VPU accessible address from any valid virtual address
* @param ptr [IN] : the valid (possibly offseted) virtual address inside
*                   a valid allocated mem-mapped DMABuf.
*                   the corresponding DMABuf fd should have been imported by vpusmm_import_dmabuf()
* @return VPU-side address corresponds to the input virtual address
*         or zero if the input pointer is not within a valid mem-mapped
*         buffer object.
*/
unsigned long vpusmm_ptr_to_vpu(void * ptr);

/**  get physical address from any valid virtual address
* @param ptr [IN] : any valid virtual address of user-space, not necessary comes from an imported DMABuf
* @return physical address corresponds to the input virtual address
*         or zero if the virtual address is not valid.
*
* inside kernel module, this function walks current's page tables to get physical address.
* this function has following drawbacks:
* 1. cannot differentiate whether the backed physical memory comes from DMABuf or
*    a normal system memory, so it cannot guarantee the physical memory is suitable for VPU access
* 2. it cannot guarantee the physical memory is still pinned in page frame when the address is used
*    later on.
* 3. it returns physical addresses in CPU's viewpoint, which is in general not equal to device address
*    (possible offset is not included), so device usually cannot use this address directly.
*
* so import() & ptr_to_vpu() would be a better choice.
*/
unsigned long vpusmm_ptr_to_phys(void * ptr);

#ifdef __cplusplus
}
#endif
#endif
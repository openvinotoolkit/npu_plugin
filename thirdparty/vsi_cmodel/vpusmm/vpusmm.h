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


/** vpu-shared memory type
*/
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

/**  Select which slice VPU shared buffer to be allocated, if this function is not called to specify slice index, 
*          the default slice is 0.
* @param slice_idx [IN]: index of slice, starting from 0. It indicates from which slice VPU shared buffer 
*          will be allocated.
* @return 0 indicates success, -1 indicates failure.
*/
int vpusmm_select_vpu(int slice_idx);

/** allocate a VPU-shared buffer object for multi slice
 * @param size [IN]: size in bytes, must be multiple of page size, otherwise mmap syscall may fail
 * @param type [IN]: vpu-shared memory type
 * @param slice_idx [IN]: index of slice, starting from 0. It indicates from which slice VPU shared buffer will be allocated
 * @return: the file descriptor of the underlying vpu-shared buffer object
 *          or <0 if failed
 *
 * user is supposed to call standard UNIX API close() on the fd
 * when not using it anymore, failing doing so will cause leakage
 * during runtime but will be cleaned when application process
 * ceased execution.
 *
 */
int vpurm_alloc_dmabuf(unsigned long size, enum VPUSMMType type, int slice_idx);

/**  Import an external DMABuf object into VPU device for accessing on VPU side for multi slice
* @param dmabuf_fd [IN] : the DMABuf fd exported by other driver(codec/camera/...) or our driver
* @param vpu_access [IN] : how VPU is going to access the buffer (readonly or writeonly or readwrite)
*                          application should provide this param as accurate as possible, use VPU_DEFAULT
*                          when not sure, this mainly affect cache coherency
* @param slice_idx [IN]: index of slice, starting from 0. It indicates from which slice VPU shared buffer will be allocated
* @return a non-zero VPU-side address corresponds to the DMABuf which suitable for VPU to access.
*         zero if the DMABuf is not suitable for VPU to access (for example, due to not physical contiguous)
*         in this case, user should allocate another DMABuf with VPUSMM and do copy by CPU
*
* import operation is vital for VPU to access the buffer because:
*   1. necessary mapping resource is setup and maintained during import for VPU to access the buffer
*   2. the physical pages is pinned during import so no page fault will happen
*   3. kernel rb-tree is setup and maintained so virt-addr can be translated into device address
*/
unsigned long vpurm_import_dmabuf(int dmabuf_fd, enum VPUAccessType vpu_access, int slice_idx);

/**  Unimport an imported external DMABuf when VPU side no longer accesses it from multi slice
* @param dmabuf_fd [IN] : the DMABuf fd to be unimported
* @param slice_idx [IN]: index of slice, starting from 0. It indicates from which slice VPU shared buffer will be allocated
* after this function call, the reference to the DMABuf will be freed and the VPU address should not
* be considered to be valid any more, so call this function only after VPU-side access has done.
*/
void vpurm_unimport_dmabuf(int dmabuf_fd, int slice_idx);

/**  get VPU accessible address from any valid virtual address
* @param ptr [IN] : the valid (possibly offseted) virtual address inside
*                   a valid allocated mem-mapped DMABuf.
*                   the corresponding DMABuf fd should have been imported by vpusmm_import_dmabuf()
*@param slice_idx [IN]: index of slice, starting from 0. It indicates from which slice VPU shared buffer will be allocated
* @return VPU-side address corresponds to the input virtual address
*         or zero if the input pointer is not within a valid mem-mapped
*         buffer object.
*/
unsigned long vpurm_ptr_to_vpu(void * ptr, int slice_idx);

#ifdef __cplusplus
}
#endif
#endif
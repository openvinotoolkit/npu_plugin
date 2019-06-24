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

#ifndef _LINUX_DMA_DIRECTION_H
#define _LINUX_DMA_DIRECTION_H

/** Data access direction hint for cache coherent maintainance
*/
enum dma_data_direction {
    DMA_BIDIRECTIONAL = 0,    /**< device(VPU) will do both read & write */
    DMA_TO_DEVICE = 1,        /**< device(VPU) will only read data */
    DMA_FROM_DEVICE = 2,      /**< device(VPU) will only write data */
    DMA_NONE = 3,             /**< not used */
};

#endif


/** allocate a VPU-shared buffer object
 * @param size [IN]: size in bytes
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
* @param direction [IN] :
* @return a non-zero VPU-side address corresponds to the DMABuf which suitable for VPU to access.
*         zero if the DMABuf is not suitable for VPU to access (for example, due to not physical contiguous)
*         in this case, user should allocate another DMABuf with VPUSMM and do copy by CPU
*/
unsigned long vpusmm_import_dmabuf(int dmabuf_fd, enum dma_data_direction direction);


/**  Import an external DMABuf object into VPU device for accessing on VPU side
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


#ifdef __cplusplus
}
#endif
#endif